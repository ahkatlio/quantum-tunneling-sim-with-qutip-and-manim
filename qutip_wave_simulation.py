import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich.text import Text
from scipy.integrate import trapezoid

plt.rcParams['text.usetex'] = False
class QuantumWavePacket:
    def __init__(self, sigma=1.2, x_start=-20.0, x_end=20.0, dx=0.1, x0=-10.0, k0=2.0, 
                 height=1.5, width=1.0, dt=0.05, nt=400, auto_adjust_nt=True):
        self.console = Console()
        
        self.sigma = sigma
        self.x_start = x_start
        self.x_end = x_end
        self.dx = dx
        self.x0 = x0
        self.k0 = k0
        self.height = height
        self.width = width
        self.dt = dt
        self.nt = nt
        self.auto_adjust_nt = auto_adjust_nt
        self.optimal_nt = None
        
        self.data_dir = Path(__file__).parent / "Data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.points = int((x_end - x_start) / dx)
        self.x = np.linspace(x_start, x_end, self.points)
        
        normalization = 1.0 / ((2 * np.pi * sigma**2)**(1/4))
        self.psi = normalization * np.exp(-((self.x - x0)/(2*sigma))**2) * np.exp(1j * k0 * self.x)
        
        self.prob = abs(self.psi)**2
        norm = np.sqrt(np.trapezoid(self.prob, self.x))
        self.psi /= norm
        self.prob = abs(self.psi)**2
        
        self.initial_state = Qobj(self.psi.reshape(-1, 1))
        
        self.v = np.zeros(self.points, dtype=complex)
        mid_value = int(self.points/2)
        end_points = int(width/dx) + mid_value
        self.v[mid_value:end_points] = height
        
        laplace = np.zeros((self.points, self.points))
        
        for i in range(1, self.points - 1):
            laplace[i, i-1] = 1
            laplace[i, i] = -2
            laplace[i, i+1] = 1
        
        laplace[0, 0] = -2
        laplace[0, 1] = 1
        laplace[self.points-1, self.points-2] = 1
        laplace[self.points-1, self.points-1] = -2  
        
        H_matrix = (-1/(2 * self.dx**2) * laplace) + np.diag(self.v)
        
        self.H = Qobj(H_matrix)
        self.psi_qobj = Qobj(self.psi.reshape(-1, 1))
        self.psi_qobj = self.psi_qobj.unit()
        
        I = np.eye(self.points)
        Matrix_1 = I + (dt/2j) * H_matrix
        Matrix_2 = np.linalg.inv(I - (dt/2j) * H_matrix)
        self.U = Qobj(np.matmul(Matrix_2, Matrix_1))
        
        self.x_op = Qobj(np.diag(self.x))
        
        p_matrix = np.zeros((self.points, self.points), dtype=complex)
        
        for i in range(1, self.points - 1):
            p_matrix[i, i-1] = -1j / (2 * self.dx)
            p_matrix[i, i+1] = 1j / (2 * self.dx)
        
        p_matrix[0, 0] = -1j / self.dx
        p_matrix[0, 1] = 1j / self.dx
        
        p_matrix[self.points-1, self.points-2] = -1j / self.dx
        p_matrix[self.points-1, self.points-1] = 1j / self.dx
        
        p_matrix = 0.5 * (p_matrix + p_matrix.conj().T)
        
        self.p_op = Qobj(p_matrix)
        
        self._display_parameters()
        self._verify_initial_uncertainty()
    
    def _display_parameters(self):
        table = Table(title="Quantum Wave Packet Simulation Parameters")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Grid Points", str(self.points))
        table.add_row("Domain", f"[{self.x_start:.1f}, {self.x_end:.1f}]")
        table.add_row("Grid Spacing (dx)", f"{self.dx:.3f}")
        table.add_row("Time Step (dt)", f"{self.dt:.3f}")
        table.add_row("Number of Steps", str(self.nt))
        table.add_row("Initial Position ($x_0$)", f"{self.x0:.2f}")
        table.add_row("Initial Momentum ($k_0$)", f"{self.k0:.2f}")
        table.add_row("Wave Packet Width ($\\sigma$)", f"{self.sigma:.2f}")
        table.add_row("Barrier Height", f"{self.height:.2f}")
        table.add_row("Barrier Width", f"{self.width:.2f}")
        table.add_row("Initial Energy", f"{self.k0**2/2:.2f}")
        
        self.console.print(Panel(table, title="Simulation Setup", border_style="blue"))
    
    def _verify_initial_uncertainty(self):
        exp_x = expect(self.x_op, self.psi_qobj)
        exp_p = expect(self.p_op, self.psi_qobj)
        exp_x2 = expect(self.x_op * self.x_op, self.psi_qobj)
        exp_p2 = expect(self.p_op * self.p_op, self.psi_qobj)
        
        var_x = np.real(exp_x2 - exp_x**2)
        var_p = np.real(exp_p2 - exp_p**2)
        
        delta_x = np.sqrt(max(0, var_x))
        delta_p = np.sqrt(max(0, var_p))
        uncertainty_product = delta_x * delta_p
        
        verification_table = Table(title="Initial State Verification")
        verification_table.add_column("Property", style="cyan")
        verification_table.add_column("Value", style="yellow")
        verification_table.add_column("Expected", style="green")
        
        verification_table.add_row("$\\langle x \\rangle$", f"{np.real(exp_x):.3f}", f"{self.x0:.3f}")
        verification_table.add_row("$\\langle p \\rangle$", f"{np.real(exp_p):.3f}", f"{self.k0:.3f}")
        verification_table.add_row("$\\Delta x$", f"{delta_x:.3f}", f"{self.sigma:.3f}")
        verification_table.add_row("$\\Delta p$", f"{delta_p:.3f}", f"{1/(2*self.sigma):.3f}")
        verification_table.add_row("$\\Delta x \\cdot \\Delta p$", f"{uncertainty_product:.3f}", "≥ 0.500")

        status = "✅ CORRECT" if uncertainty_product >= 0.49 else "❌ INCORRECT"
        verification_table.add_row("Uncertainty Check", status, "≥ \\hbar/2")
        
        self.console.print(verification_table)
        
        return uncertainty_product
    
    def find_optimal_time(self):
        self.console.print(f"\n[yellow]Phase 1: Finding optimal stopping time...[/yellow]")
        
        current_state = self.psi_qobj
        uncertainty_products = []
        
        for i in range(self.nt):
            exp_x = expect(self.x_op, current_state)
            exp_p = expect(self.p_op, current_state)
            exp_x2 = expect(self.x_op * self.x_op, current_state)
            exp_p2 = expect(self.p_op * self.p_op, current_state)
            
            var_x = np.real(exp_x2 - exp_x**2)
            var_p = np.real(exp_p2 - exp_p**2)
            
            delta_x = np.sqrt(max(0, var_x))
            delta_p = np.sqrt(max(0, var_p))
            uncertainty_product = delta_x * delta_p
            uncertainty_products.append(uncertainty_product)
            
            if i > 10:
                if len(uncertainty_products) >= 5:
                    if (i >= 3 and 
                        uncertainty_products[-1] < uncertainty_products[-2] and 
                        uncertainty_products[-2] < uncertainty_products[-3]):
                        
                        recent_window = min(20, len(uncertainty_products))
                        recent_uncertainties = uncertainty_products[-recent_window:]
                        max_uncertainty = max(recent_uncertainties)
                        max_index = len(uncertainty_products) - recent_window + recent_uncertainties.index(max_uncertainty)
                        
                        optimal_time = max_index * self.dt
                        optimal_nt = max_index
                        
                        self.console.print(f"[green]✓ Found optimal stopping time: {optimal_time:.2f}s (step {optimal_nt})[/green]")
                        return optimal_nt, optimal_time
                        
            if i < self.nt - 1:
                current_state = self.U * current_state
        
        self.console.print(f"[yellow]No peak found, using full simulation time[/yellow]")
        return self.nt, self.nt * self.dt

    def evolve_step_by_step(self):
        if self.auto_adjust_nt:
            optimal_nt, optimal_time = self.find_optimal_time()
            self.optimal_nt = optimal_nt
            show_time = max(optimal_time - 8.0, optimal_time * 0.5)
            actual_nt = int(show_time / self.dt)
            self.console.print(f"\n[cyan]Phase 2: Running optimized simulation for {actual_nt} steps ({actual_nt * self.dt:.2f}s)...[/cyan]")
            self.console.print(f"[yellow]   (Optimal peak at {optimal_time:.2f}s, showing until {show_time:.2f}s for better visualization)[/yellow]")
        else:
            actual_nt = self.nt
            self.console.print(f"\n[cyan]Evolving wave packet for {actual_nt} steps...[/cyan]")
        
        states = [self.psi_qobj]
        times = [0.0]
        probabilities = []
        expectations_x = []
        expectations_p = []
        variances_x = []
        variances_p = []
        transmission_prob = []
        reflection_prob = []
        barrier_prob = []
        absorbed_prob = []
        conservation_error = []
        uncertainty_products = []
        
        current_state = self.psi_qobj
        
        psi_array = current_state.full().flatten()
        prob_density = np.abs(psi_array)**2
        prob_total = np.trapezoid(prob_density, self.x)
        if prob_total > 0:
            prob_density = prob_density / prob_total
        probabilities.append(prob_density.tolist())
        
        exp_x = expect(self.x_op, current_state)
        exp_p = expect(self.p_op, current_state)
        exp_x2 = expect(self.x_op * self.x_op, current_state)
        exp_p2 = expect(self.p_op * self.p_op, current_state)
        
        expectations_x.append(np.real(exp_x))
        expectations_p.append(np.real(exp_p))
        
        var_x = np.real(exp_x2 - exp_x**2)
        var_p = np.real(exp_p2 - exp_p**2)
        variances_x.append(max(0, var_x))
        variances_p.append(max(0, var_p))
        
        delta_x = np.sqrt(max(0, var_x))
        delta_p = np.sqrt(max(0, var_p))
        uncertainty_product = delta_x * delta_p
        uncertainty_products.append(uncertainty_product)
        
        barrier_center = (self.x_start + self.x_end) / 2
        barrier_left = barrier_center - self.width/2
        barrier_right = barrier_center + self.width/2
        
        refl_indices = self.x < barrier_left
        trans_indices = self.x > barrier_right
        barrier_indices = (self.x >= barrier_left) & (self.x <= barrier_right)
        
        refl_prob = trapezoid(prob_density[refl_indices], self.x[refl_indices]) if np.any(refl_indices) else 0.0
        trans_prob = trapezoid(prob_density[trans_indices], self.x[trans_indices]) if np.any(trans_indices) else 0.0
        barrier_prob_val = trapezoid(prob_density[barrier_indices], self.x[barrier_indices]) if np.any(barrier_indices) else 0.0
        absorbed_prob_direct = 0.0
        total_prob_direct = refl_prob + trans_prob + barrier_prob_val + absorbed_prob_direct
        
        transmission_prob.append(float(f"{trans_prob:.12f}"))
        reflection_prob.append(float(f"{refl_prob:.12f}"))
        barrier_prob.append(float(f"{barrier_prob_val:.12f}"))
        absorbed_prob.append(float(f"{absorbed_prob_direct:.12f}"))
        conservation_error.append(float(f"{abs(total_prob_direct - 1.0):.12f}"))
        
        for i in track(range(1, actual_nt), description="Time evolution"):
            current_state = self.U * current_state
            states.append(current_state)
            times.append(i * self.dt)
            
            psi_array = current_state.full().flatten()
            prob_density = np.abs(psi_array)**2
            
            prob_total = np.trapezoid(prob_density, self.x)
            if prob_total > 0:
                prob_density = prob_density / prob_total
            
            probabilities.append(prob_density.tolist())
            
            exp_x = expect(self.x_op, current_state)
            exp_p = expect(self.p_op, current_state)
            exp_x2 = expect(self.x_op * self.x_op, current_state)
            exp_p2 = expect(self.p_op * self.p_op, current_state)
            
            expectations_x.append(np.real(exp_x))
            expectations_p.append(np.real(exp_p))
            
            var_x = np.real(exp_x2 - exp_x**2)
            var_p = np.real(exp_p2 - exp_p**2)
            variances_x.append(max(0, var_x))
            variances_p.append(max(0, var_p))
            
            delta_x = np.sqrt(max(0, var_x))
            delta_p = np.sqrt(max(0, var_p))
            uncertainty_product = delta_x * delta_p
            uncertainty_products.append(uncertainty_product)
            
            barrier_center = (self.x_start + self.x_end) / 2
            barrier_left = barrier_center - self.width/2
            barrier_right = barrier_center + self.width/2
            
            refl_indices = self.x < barrier_left
            trans_indices = self.x > barrier_right
            barrier_indices = (self.x >= barrier_left) & (self.x <= barrier_right)
            
            refl_prob = trapezoid(prob_density[refl_indices], self.x[refl_indices]) if np.any(refl_indices) else 0.0
            trans_prob = trapezoid(prob_density[trans_indices], self.x[trans_indices]) if np.any(trans_indices) else 0.0
            barrier_prob_val = trapezoid(prob_density[barrier_indices], self.x[barrier_indices]) if np.any(barrier_indices) else 0.0
            
            absorbed_prob_direct = 0.0
            
            total_prob_direct = refl_prob + trans_prob + barrier_prob_val + absorbed_prob_direct
            
            transmission_prob.append(float(f"{trans_prob:.12f}"))
            reflection_prob.append(float(f"{refl_prob:.12f}"))
            barrier_prob.append(float(f"{barrier_prob_val:.12f}"))
            
            absorbed_prob.append(float(f"{absorbed_prob_direct:.12f}"))
            conservation_error.append(float(f"{abs(total_prob_direct - 1.0):.12f}"))
        
        return {
            'states': states,
            'times': times,
            'probabilities': probabilities,
            'expectations_x': expectations_x,
            'expectations_p': expectations_p,
            'variances_x': variances_x,
            'variances_p': variances_p,
            'transmission_prob': transmission_prob,
            'reflection_prob': reflection_prob,
            'barrier_prob': barrier_prob,
            'absorbed_prob': absorbed_prob,
            'conservation_error': conservation_error,
            'uncertainty_products': uncertainty_products
        }
    
    def analyze_results(self, evolution_data):
        self.console.print("\n[bold blue]Analyzing results and creating visualizations...[/bold blue]")
        
        states = evolution_data['states']
        times = evolution_data['times']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        initial_prob = np.abs(states[0].full().flatten())**2
        final_prob = np.abs(states[-1].full().flatten())**2
        
        ax1.plot(self.x, initial_prob, 'b-', label='Initial', linewidth=2)
        ax1.plot(self.x, final_prob, 'r-', label='Final', linewidth=2)
        
        V_real = np.real(self.v)
        if np.max(V_real) > 0:
            barrier_scale = np.max(initial_prob) / np.max(V_real)
            ax1.plot(self.x, V_real * barrier_scale, 'k--', label='Barrier', alpha=0.7, linewidth=2)
        
        ax1.set_xlabel(r'Position $x$')
        ax1.set_ylabel(r'Probability Density $|\psi(x)|^2$')
        ax1.set_title(r'Initial vs Final Probability Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(times, evolution_data['expectations_x'], 'g-', linewidth=2)
        barrier_center = (self.x_start + self.x_end) / 2
        ax2.axhline(y=barrier_center, color='k', linestyle='--', alpha=0.7, label='Barrier Center')
        ax2.set_xlabel(r'Time $t$')
        ax2.set_ylabel(r'$\langle x \rangle$')
        ax2.set_title(r'Average Position vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        total_prob = [t + r + b for t, r, b in zip(evolution_data['transmission_prob'],
                                                  evolution_data['reflection_prob'],
                                                  evolution_data['barrier_prob'])]
        
        ax3.plot(times, evolution_data['transmission_prob'], 'b-', label='Transmission (T)', linewidth=2)
        ax3.plot(times, evolution_data['reflection_prob'], 'r-', label='Reflection (R)', linewidth=2)
        ax3.plot(times, evolution_data['barrier_prob'], 'g--', label='Barrier (B)', linewidth=2)
        ax3.plot(times, evolution_data['absorbed_prob'], 'm:', label='Absorbed (A)', linewidth=2)
        
        total_calculated = [r + t + b + a for r, t, b, a in zip(
            evolution_data['reflection_prob'], evolution_data['transmission_prob'],
            evolution_data['barrier_prob'], evolution_data['absorbed_prob'])]
        ax3.plot(times, total_calculated, 'k-', label='R+T+B+A', linewidth=1, alpha=0.7)
        
        ax3.set_xlabel(r'Time $t$')
        ax3.set_ylabel(r'Probability')
        ax3.set_title(r'Probability Components vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.05, 1.05)
        
        std_x = np.sqrt(evolution_data['variances_x'])
        std_p = np.sqrt(evolution_data['variances_p'])
        uncertainty_product = std_x * std_p
        
        valid_indices = []
        for i, t in enumerate(times):
            if i < len(evolution_data['probabilities']):
                prob_density = np.array(evolution_data['probabilities'][i])
                total_prob = np.trapz(prob_density, self.x)
                if total_prob > 0.05 and uncertainty_product[i] > 0.01:  
                    valid_indices.append(i)
        
        if valid_indices:
            valid_times = [times[i] for i in valid_indices]
            valid_uncertainty = [uncertainty_product[i] for i in valid_indices]
            
            ax4.plot(valid_times, valid_uncertainty, 'm-', linewidth=2, label=r'$\Delta x \cdot \Delta p$')
            ax4.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label=r'$\hbar/2$ minimum')
            
            if len(valid_times) < len(times):
                cutoff_time = valid_times[-1] if valid_times else 0
                ax4.axvline(x=cutoff_time, color='b', linestyle=':', alpha=0.7, 
                          label=f'Wave exits domain (t~{cutoff_time:.1f})')
        else:
            ax4.text(0.5, 0.5, 'Wave exited domain\nbefore significant evolution', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_xlabel(r'Time $t$')
        ax4.set_ylabel(r'$\Delta x \cdot \Delta p$')
        ax4.set_title(r'Heisenberg Uncertainty Principle')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.data_dir / 'wave_packet_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.console.print(f"[green]Analysis plot saved as '{plot_path}'[/green]")
        
        plt.show()
        
        self._display_final_results(evolution_data)
        
        return len(uncertainty_product) > 0 and uncertainty_product[-1] or 0
    
    def _display_final_results(self, evolution_data):
        final_transmission = evolution_data['transmission_prob'][-1]
        final_reflection = evolution_data['reflection_prob'][-1]
        final_barrier = evolution_data['barrier_prob'][-1]
        final_absorbed = evolution_data['absorbed_prob'][-1]
        final_conservation_error = evolution_data['conservation_error'][-1]
        total_final = final_transmission + final_reflection + final_barrier + final_absorbed
        
        final_uncertainty = (np.sqrt(evolution_data['variances_x'][-1]) * 
                           np.sqrt(evolution_data['variances_p'][-1]))
        
        final_prob_density = np.array(evolution_data['probabilities'][-1])
        final_total_prob = np.trapz(final_prob_density, self.x)
        uncertainty_meaningful = final_total_prob > 0.05
        
        results_table = Table(title="Final Simulation Results")
        results_table.add_column("Property", style="cyan")
        results_table.add_column("Value", style="green")
        results_table.add_column("Percentage/Status", style="yellow")
        
        results_table.add_row("Final Time", f"{evolution_data['times'][-1]:.2f}", f"t = {evolution_data['times'][-1]:.1f}")
        results_table.add_row("Transmission (T)", f"{final_transmission:.8f}", f"{final_transmission*100:.3f}%")
        results_table.add_row("Reflection (R)", f"{final_reflection:.8f}", f"{final_reflection*100:.3f}%")
        results_table.add_row("Barrier (B)", f"{final_barrier:.8f}", f"{final_barrier*100:.3f}%")
        results_table.add_row("Absorbed (A)", f"{final_absorbed:.8f}", f"{final_absorbed*100:.3f}%")
        results_table.add_row("Total (R+T+B+A)", f"{total_final:.8f}", f"{total_final*100:.3f}%")
        
        if final_conservation_error < 1e-6:
            conservation_status = "[green]Excellent[/green]"
        elif final_conservation_error < 1e-4:
            conservation_status = "[yellow]Good[/yellow]"
        else:
            conservation_status = "[red]Poor[/red]"
        
        results_table.add_row("Conservation Error", f"{final_conservation_error:.2e}", conservation_status)
        
        initial_R = evolution_data['reflection_prob'][0]
        initial_T = evolution_data['transmission_prob'][0]
        initial_B = evolution_data['barrier_prob'][0]
        initial_A = evolution_data['absorbed_prob'][0]
        
        results_table.add_row("", "", "")
        results_table.add_row("Initial R", f"{initial_R:.8f}", "Expected: ~1.0")
        results_table.add_row("Initial T", f"{initial_T:.8f}", "Expected: ~0.0")
        results_table.add_row("Initial B", f"{initial_B:.8f}", "Expected: ~0.0")
        results_table.add_row("Initial A", f"{initial_A:.8f}", "Expected: ~0.0")
        
        results_table.add_row(r"Final Δx · Δp", f"{final_uncertainty:.3f}", 
                            "✅ Valid" if uncertainty_meaningful and final_uncertainty >= 0.49 
                            else "N/A (absorbed)" if not uncertainty_meaningful 
                            else "❌ Invalid")
        
        self.console.print(Panel(results_table, title="Simulation Complete", border_style="green"))
        
        self.console.print("\n[bold blue]Physical Interpretation:[/bold blue]")
        
        if final_transmission > 0.1:
            self.console.print(f" [green]Significant tunneling: {final_transmission*100:.1f}% transmission[/green]")
        else:
            self.console.print(f" [yellow]Limited tunneling: {final_transmission*100:.1f}% transmission[/yellow]")
        
        if final_absorbed > 0.1:
            self.console.print(f" [blue]Wave absorption: {final_absorbed*100:.1f}% of wave was absorbed at boundaries[/blue]")
        
        initial_energy = self.k0**2 / 2
        self.console.print(f" Initial kinetic energy: {initial_energy:.2f}")
        self.console.print(f" Barrier height: {self.height:.2f}")
        
        if initial_energy > self.height:
            self.console.print(" [green]Classical prediction: Should mostly transmit[/green]")
        else:
            self.console.print(" [red]Classical prediction: Should be completely reflected[/red]")
    
    def save_animation_data(self, evolution_data, filename='animation_data.json'):
        self.console.print(f"\n[bold blue]Saving animation data for Manim...[/bold blue]")
        
        animation_data = {
            'x_grid': self.x.tolist(),
            'potential': np.real(self.v).tolist(),
            'times': evolution_data['times'],
            'probabilities': evolution_data['probabilities'],
            'barrier_position': float((self.x_start + self.x_end) / 2),
            'barrier_width': float(self.width),
            'barrier_height': float(self.height),
            'x_start': float(self.x_start),
            'x_end': float(self.x_end),
            'expectations_x': evolution_data['expectations_x'],
            'expectations_p': evolution_data['expectations_p'],
            'variances_x': evolution_data['variances_x'],
            'variances_p': evolution_data['variances_p'],
            'transmission_prob': evolution_data['transmission_prob'],
            'reflection_prob': evolution_data['reflection_prob'],
            'barrier_prob': evolution_data['barrier_prob'],
            'absorbed_prob': evolution_data['absorbed_prob'],
            'conservation_error': evolution_data['conservation_error']
        }
        
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(animation_data, f, indent=2)
        
        self.console.print(f"[green]Animation data saved to '{filepath}'[/green]")
        return animation_data


def main():
    console = Console()
    
    title = Text("Quantum Wave Packet Tunneling Simulation", style="bold magenta")
    console.print(Panel(title, border_style="bright_blue"))    
    wave = QuantumWavePacket(
        sigma=1.2,
        x_start=-40.0,
        x_end=40.0,
        dx=0.1,
        x0=-8.0,
        k0=2.0,
        height=1.5,
        width=1.0,
        dt=0.05,
        nt=1000
    )
    
    evolution_data = wave.evolve_step_by_step()
    
    final_uncertainty = wave.analyze_results(evolution_data)
    
    animation_data = wave.save_animation_data(evolution_data)
    
    console.print(f"\n[bold green]Simulation completed successfully![/bold green]")
    console.print(f" Final uncertainty product: {final_uncertainty:.3f}")
    console.print(f" Data saved for Manim visualization")
    console.print(f" Analysis plots generated")
    
    return wave, evolution_data, animation_data


if __name__ == "__main__":
    wave, evolution_data, animation_data = main()
