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
                 height=1.5, width=1.0, dt=0.05, nt=400):
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
        
        self.data_dir = Path(__file__).parent / "Data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.points = int((x_end - x_start) / dx)
        self.x = np.linspace(x_start, x_end, self.points)
        
        normalization = 1.0 / ((2 * np.pi * sigma**2)**(1/4))
        self.psi = normalization * np.exp(-((self.x - x0)/(2*sigma))**2) * np.exp(1j * k0 * self.x)
        
        self.prob = abs(self.psi)**2
        norm = np.sqrt(np.trapezoid(self.prob, self.x))
        self.prob /= norm**2
        self.psi /= norm
        
        self.initial_state = Qobj(self.psi.reshape(-1, 1))
        self.initial_state = self.initial_state.unit()
        
        self.v = np.zeros(self.points, dtype=complex)
        mid_value = int(self.points/2)
        end_points = int(width/dx) + mid_value
        self.v[mid_value:end_points] = height
        
        absorb_length = int(self.points * 0.2)
        absorb_strength = 50.0
        
        for i in range(absorb_length):
            idx = self.points - absorb_length + i
            ramp = (i / absorb_length)**2
            self.v[idx] += -1j * absorb_strength * ramp

        for i in range(absorb_length):
            ramp = ((absorb_length - 1 - i) / absorb_length)**2
            self.v[i] += -1j * absorb_strength * ramp
        
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
        table.add_row("Initial Position ($x_\\theta$)", f"{self.x0:.2f}")
        table.add_row("Initial Momentum ($k_\\theta$)", f"{self.k0:.2f}")
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

        status = "✓ CORRECT" if uncertainty_product >= 0.49 else "✗ INCORRECT"
        verification_table.add_row("Uncertainty Check", status, "≥ \\hbar/2")
        
        self.console.print(verification_table)
        
        return uncertainty_product
    
    def evolve_step_by_step(self):
        self.console.print(f"\n[cyan]Evolving wave packet for {self.nt} time steps...[/cyan]")
        
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
        
        current_state = self.psi_qobj
        
        for i in track(range(self.nt), description="Time evolution"):
            t = i * self.dt
            
            psi_array = current_state.full().flatten()
            prob_density = np.abs(psi_array)**2
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
            
            barrier_center = (self.x_start + self.x_end) / 2
            barrier_left = barrier_center - self.width/2
            barrier_right = barrier_center + self.width/2
            
            absorb_length = int(self.points * 0.1)
            left_absorb_end = self.x[absorb_length]
            right_absorb_start = self.x[-absorb_length]
            
            trans_indices = (self.x > barrier_right) & (self.x < right_absorb_start)
            trans_prob = trapezoid(prob_density[trans_indices], self.x[trans_indices]) if np.any(trans_indices) else 0.0
            
            refl_indices = (self.x < barrier_left) & (self.x > left_absorb_end)
            refl_prob = trapezoid(prob_density[refl_indices], self.x[refl_indices]) if np.any(refl_indices) else 0.0
            
            barrier_indices = (self.x >= barrier_left) & (self.x <= barrier_right)
            barrier_prob_val = trapezoid(prob_density[barrier_indices], self.x[barrier_indices]) if np.any(barrier_indices) else 0.0
            
            transmission_prob.append(trans_prob)
            reflection_prob.append(refl_prob)
            barrier_prob.append(barrier_prob_val)
            
            if i < self.nt - 1:
                current_state = self.U * current_state
                states.append(current_state)
                times.append((i + 1) * self.dt)
        
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
            'barrier_prob': barrier_prob
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
        
        ax3.plot(times, evolution_data['transmission_prob'], 'b-', label='Transmission', linewidth=2)
        ax3.plot(times, evolution_data['reflection_prob'], 'r-', label='Reflection', linewidth=2)
        ax3.plot(times, evolution_data['barrier_prob'], 'g--', label='Barrier', linewidth=2)
        ax3.plot(times, total_prob, 'k:', label='Total', linewidth=2)
        
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
        total_final = final_transmission + final_reflection + final_barrier
        
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
        results_table.add_row("Transmission Probability", f"{final_transmission:.4f}", f"{final_transmission*100:.1f}%")
        results_table.add_row("Reflection Probability", f"{final_reflection:.4f}", f"{final_reflection*100:.1f}%")
        results_table.add_row("Barrier Probability", f"{final_barrier:.4f}", f"{final_barrier*100:.1f}%")
        results_table.add_row("Total Probability", f"{total_final:.4f}", f"{total_final*100:.1f}%")
        
        if abs(total_final - 1.0) < 0.01:
            conservation_status = "[green]Excellent[/green]"
        elif abs(total_final - 1.0) < 0.05:
            conservation_status = "[yellow]Good[/yellow]"
        else:
            conservation_status = "[red]Poor[/red]"
        
        results_table.add_row("Probability Conservation", f"{total_final:.4f}", conservation_status)
        
        absorbed_prob = 1.0 - total_final
        results_table.add_row("Absorbed Probability", f"{absorbed_prob:.4f}", f"{absorbed_prob*100:.1f}%")
        
        results_table.add_row("Final Δx⋅Δp", f"{final_uncertainty:.3f}", 
                            "✓ Valid" if uncertainty_meaningful and final_uncertainty >= 0.49 
                            else "N/A (absorbed)" if not uncertainty_meaningful 
                            else "✗ Invalid")
        
        self.console.print(Panel(results_table, title="Simulation Complete", border_style="green"))
        
        self.console.print("\n[bold blue]Physical Interpretation:[/bold blue]")
        
        if final_transmission > 0.1:
            self.console.print(f"• [green]Significant tunneling: {final_transmission*100:.1f}% transmission[/green]")
        else:
            self.console.print(f"• [yellow]Limited tunneling: {final_transmission*100:.1f}% transmission[/yellow]")
        
        if absorbed_prob > 0.1:
            self.console.print(f"• [blue]Wave absorption: {absorbed_prob*100:.1f}% of wave was absorbed at boundaries[/blue]")
        
        initial_energy = self.k0**2 / 2
        self.console.print(f"• Initial kinetic energy: {initial_energy:.2f}")
        self.console.print(f"• Barrier height: {self.height:.2f}")
        
        if initial_energy > self.height:
            self.console.print("• [green]Classical prediction: Should mostly transmit[/green]")
        else:
            self.console.print("• [red]Classical prediction: Should be completely reflected[/red]")
    
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
            'barrier_prob': evolution_data['barrier_prob']
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
        x_start=-20.0,
        x_end=20.0,
        dx=0.1,
        x0=-8.0,
        k0=2.0,
        height=1.5,
        width=1.0,
        dt=0.05,
        nt=400
    )
    
    evolution_data = wave.evolve_step_by_step()
    
    final_uncertainty = wave.analyze_results(evolution_data)
    
    animation_data = wave.save_animation_data(evolution_data)
    
    console.print(f"\n[bold green]Simulation completed successfully![/bold green]")
    console.print(f"• Final uncertainty product: {final_uncertainty:.3f}")
    console.print(f"• Data saved for Manim visualization")
    console.print(f"• Analysis plots generated")
    
    return wave, evolution_data, animation_data


if __name__ == "__main__":
    wave, evolution_data, animation_data = main()
