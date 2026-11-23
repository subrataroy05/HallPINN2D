import math
import torch
import torch.nn as nn


VETHETA_MAX = 2000.0


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D cylindrical Hall Effect Thruster simulation.

    v3: v26 reduced architecture with direct free ion density, free
    interior potential shape, solved Vix, and solved Te.
    output_dim = 10; old v26/v3 checkpoints are intentionally incompatible.

    Network outputs (10 values):
        raw[:,0]: free phi interior shape in tanh(raw0), scaled by A=25
        raw[:,1]: direct ni log-amplitude with hard anode Dirichlet
        raw[:,2]: transported Vex = A_ex*tanh(raw2)
        raw[:,3]: Vir amplitude
        raw[:,4]: free nn depletion exponent, referenced to the anode value
        raw[:,5]: Ver amplitude
        raw[:,6]: Vetheta amplitude
        raw[:,7]: eta -- ambipolar deviation
        raw[:,8]: Te electron temperature correction (softplus, >=0); the field
                  is Te = Te_anode + xi*softplus(raw8) -> hard Dirichlet Te(x=0)=Te_anode
        raw[:,9]: additive solved Vix correction above Bohm/inlet speed

    Forward returns 11 tensors:
        (phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, eta)
    """

    def __init__(
        self,
        input_dim=2,
        output_dim=10,
        hidden_dim=128,
        num_hidden_layers=4,
        phi_left=24.79,
        phi_right=0.0,
        x_min=0.0,
        x_max=1.0,
        r_min=1.579,
        r_max=2.237,
        delta=0.1,
        Vix_inlet=0.50,
        ni_inlet=0.14,
        nn_inlet=1.0,
        A_ex=1000.0,
        Te_base=0.25,
        Te_peak=1.0,
        Te_min=0.025 / 12.1,
        Te_init=10.0 / 12.1,
        Te_anode=2.0 / 12.1,
        x0_Te=0.40,
        sigma_x_Te=0.12,
        r0_Te=1.9,
        sigma_r_Te=0.10,
        B_max=0.020,
        x_B=0.75,
        sigma_B=0.30,
        eta_max=1.0e-4,
        phi_drop_x0_init=0.65,
        phi_drop_sigma_init=0.20,
    ):
        super().__init__()

        self.phi_left  = phi_left
        self.phi_right = phi_right
        self.x_min     = x_min
        self.x_max     = x_max
        self.r_min     = r_min
        self.r_max     = r_max
        self.delta     = delta

        self.Vix_inlet = Vix_inlet
        self.ni_inlet  = ni_inlet
        self.nn_inlet  = nn_inlet

        self.A_ex = A_ex

        self.Te_base    = Te_base
        self.Te_peak    = Te_peak
        self.Te_min     = Te_min
        self.Te_init    = Te_init
        self.Te_anode   = Te_anode
        self.x0_Te      = x0_Te
        self.sigma_x_Te = sigma_x_Te
        self.r0_Te      = r0_Te
        self.sigma_r_Te = sigma_r_Te

        self.B_max   = B_max
        self.x_B     = x_B
        self.sigma_B = sigma_B

        self.eta_max = eta_max
        self.Vetheta_max = VETHETA_MAX
        self.phi_shape_amp = 25.0
        self.Vix_anode_min = 0.05 * self.Vix_inlet
        self.ni_seed_peak = 0.060
        self.ni_seed_amp = math.log(max(self.ni_seed_peak / self.ni_inlet, 1.0))

        # phi_drop_* arguments are kept in the signature for compatibility with
        # older builders, but v3 no longer uses the rigid sigmoid drop.

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        with torch.no_grad():
            # Direct-ni warm start: raw1=0 plus the fixed seed envelope below
            # gives an HG-scale density hump at initialization while preserving
            # the hard anode density. The seed is only an initialization shape;
            # raw1 remains fully trainable and can erase or reshape it.
            self.net[-1].weight[1].zero_()
            self.net[-1].bias[1] = 0.0
            self.net[-1].bias[2] = 0.0   # original raw2 bias; Xavier weights are unchanged
            self.net[-1].bias[4] = 2.5   # free nn exponent warm-start
            self.net[-1].bias[7] = 0.0   # eta = 0 at init
            # Te uses a hard anode Dirichlet: Te = Te_anode + xi*softplus(raw8).
            # Warm-start so Te(exit, xi=1) ~= Te_init: softplus(bias8) = Te_init - Te_anode.
            self.net[-1].bias[8] = math.log(math.expm1(max(self.Te_init - self.Te_anode, 1e-6)))
            # Vix is solved, not algebraically tied to learned phi or to the
            # old full-drop closure. Use a near-zero-anode additive form:
            #   Vix = Vix_anode_min + xi*softplus(raw9).
            # Warm-start with a large exit-capable correction so Vix can reach
            # the expected ~14-16 km/s at x=1, while x=0 remains near zero
            # following Roy & Pandey's anode behavior instead of hard Bohm.
            self.net[-1].weight[9].zero_()
            self.net[-1].bias[9] = math.log(math.expm1(5.25))

    def forward(self, x, r):
        inp = torch.stack([x, r], dim=1)
        raw = self.net(inp)

        Lx = self.x_max - self.x_min
        Lr = self.r_max - self.r_min

        xi  = (x - self.x_min) / Lx
        rho = (r - self.r_min) / Lr

        wall_shape = (1.0 - self.delta) + self.delta * 4.0 * rho * (1.0 - rho)

        # --- phi: hard Dirichlet BCs with a free interior profile ---
        # The old single-sigmoid drop forced one smooth acceleration region and
        # spread the P5 voltage drop too far upstream. Use a linear hard-BC
        # baseline plus a strong bounded interior correction:
        #   phi = phi_L + (phi_R - phi_L)*xi + xi*(1-xi)*A*tanh(raw0).
        # The envelope makes phi(x=0)=phi_L and phi(x=1)=phi_R exactly for any
        # network output. A=25 gives enough margin to bend the mid-channel
        # potential above the linear baseline by the ~65 V HG requires.
        phi_base = self.phi_left + (self.phi_right - self.phi_left) * xi
        phi = phi_base + xi * (1.0 - xi) * self.phi_shape_amp * torch.tanh(raw[:, 0])

        # --- Vix: solved positive output with relaxed near-zero anode BC ---
        # The algebraic full-drop closure is no longer used as a warm baseline.
        # Roy & Pandey report axial ion velocity near zero close to the anode;
        # use 5% of the old Bohm inlet as a hard positive floor at x=0. The exit
        # remains free.
        Vix = self.Vix_anode_min + xi * torch.nn.functional.softplus(raw[:, 9])

        # --- ni: direct free positive density with hard anode Dirichlet ---
        # The previous F_i = ni_inlet*Vix_inlet*exp(raw1), ni=F_i/Vix ansatz
        # anchored the ion flux near its inlet scale and could not build flux
        # from distributed ionization. Use ni itself as the trainable positive
        # field, while keeping ni(x=0,r)=ni_inlet exactly:
        #   ni = ni_inlet * exp(seed(xi) + xi*raw1).
        # Ion continuity then determines the transported flux ni*Vix.
        ni_seed = self.ni_seed_amp * 4.0 * xi * (1.0 - xi)
        ni = self.ni_inlet * torch.exp(ni_seed + xi * raw[:, 1])

        # --- Vir (v24 original prefactor 0.005) ---
        Vir = 0.005 * xi * torch.tanh(raw[:, 3])

        # --- nn: free positive neutral density with hard anode inlet ---
        # Previous form exp(-xi*softplus(raw4)) structurally forced early
        # depletion. Use the network's local raw4 relative to its anode value at
        # the same radius. The shifted softplus is zero at x=0, nonnegative
        # everywhere, and no longer contains a monotone-in-x multiplier:
        #   nn(x=0,r) = nn_inlet exactly,
        #   0 < nn(x,r) <= nn_inlet.
        x_anode = torch.full_like(x, self.x_min)
        raw_anode = self.net(torch.stack([x_anode, r], dim=1))
        depletion_delta = raw[:, 4] - raw_anode[:, 4]
        depletion = torch.relu(
            torch.nn.functional.softplus(depletion_delta) - math.log(2.0)
        )
        nn_ = self.nn_inlet * torch.exp(-depletion)

        # --- Ver ---
        Ver = 0.008 * rho * (1.0 - rho) * torch.tanh(raw[:, 5])

        # --- Vetheta ---
        Vetheta = 4.0 * self.Vetheta_max * xi * (1.0 - xi) * torch.tanh(raw[:, 6])

        # --- Te: solved output with a HARD anode Dirichlet BC ---
        # Mirrors phi's hard-BC construction: an anode baseline (Te_anode) plus a
        # free correction multiplied by a factor (xi) that vanishes at the anode.
        # The factor xi vanishes ONLY at x=0 (not at the exit), so:
        #   Te(x=0) = Te_anode exactly  (hard Dirichlet, no soft loss needed),
        #   Te stays a free solved field for xi>0 (exit end is unconstrained).
        # softplus keeps the correction >= 0, so Te >= Te_anode > 0 everywhere.
        Te = self.Te_anode + xi * torch.nn.functional.softplus(raw[:, 8])

        # --- eta (softened tanh from v29+, KEEPS strict=True compatibility) ---
        # Kept: tanh(raw/3) softens the gradient at saturation. eta_max passed via __init__.
        eta = self.eta_max * torch.tanh(raw[:, 7] / 3.0)
        ne  = ni * (1.0 + eta)

        # Transported electron axial velocity; raw9 remains the ion channel.
        Vex = self.A_ex * torch.tanh(raw[:, 2])

        return phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, eta
