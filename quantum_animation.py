"""
Quantum Tunneling Animation with Manim
"""

import json
import numpy as np
from manim import *
from pathlib import Path

class QuantumTunneling(Scene):
    def construct(self):
        self.load_data()
        self.setup_scene()
        self.animate_tunneling()
        # self.show_results()
    
    def simple_round(self, values):
        return [round(v, 1) for v in values]
    
    def load_data(self):
        data_dir = Path(__file__).parent / "Data"
        
        with open(data_dir / "animation_data.json", "r") as f:
            self.anim_data = json.load(f)
        
        self.x_grid = np.array(self.anim_data["x_grid"])
        self.times = np.array(self.anim_data["times"])
        self.prob_frames = self.anim_data["probabilities"]
        
        self.barrier_pos = self.anim_data["barrier_position"]
        self.barrier_width = self.anim_data["barrier_width"]
        self.barrier_height = self.anim_data["barrier_height"]
        
        self.trans_prob = np.array(self.anim_data["transmission_prob"])
        self.refl_prob = np.array(self.anim_data["reflection_prob"])
        self.barrier_prob = np.array(self.anim_data["barrier_prob"])
        self.absorbed_prob = np.array(self.anim_data["absorbed_prob"])
        
        self.expectations_x = np.array(self.anim_data["expectations_x"])
        self.expectations_p = np.array(self.anim_data["expectations_p"])
        
    def setup_scene(self):
        self.axes = Axes(
            x_range=[-40, 40, 5],
            y_range=[0, 0.8, 0.1],
            x_length=12,
            y_length=6,
            axis_config={"color": WHITE},
            tips=False
        )
        
        x_label = self.axes.get_x_axis_label(Tex("Position $x$", font_size=40))
        y_label = self.axes.get_y_axis_label(Tex("$|\\psi(x,t)|^2$", font_size=36)).shift(DOWN * 0.8 + RIGHT * 0.8)
        
        left_edge = self.barrier_pos - self.barrier_width/2
        right_edge = self.barrier_pos + self.barrier_width/2
        barrier_height_scaled = min(0.6, self.barrier_height * 0.4)
        
        barrier_bottom_left = self.axes.coords_to_point(left_edge, 0)
        barrier_bottom_right = self.axes.coords_to_point(right_edge, 0)  
        barrier_top_left = self.axes.coords_to_point(left_edge, barrier_height_scaled)
        barrier_top_right = self.axes.coords_to_point(right_edge, barrier_height_scaled)
        
        self.barrier = Polygon(
            barrier_bottom_left, barrier_top_left, barrier_top_right, barrier_bottom_right,
            fill_color=RED, fill_opacity=0.7, stroke_color="#FF0000", stroke_width=8
        )
        
        barrier_label = Tex(f"Barrier\\\\$V_0 = {self.barrier_height}$", 
                           font_size=24, color=RED).next_to(self.barrier, UP, buff=0.1)
        
        # No absorber regions - clean visualization
        # absorb_length = len(self.x_grid) * 0.2
        # left_absorb_region = Rectangle(...)
        # right_absorb_region = Rectangle(...)
        # absorb_label = Tex("Absorbing\\\\Boundaries", ...)
        
        title = Tex("Quantum Tunneling Simulation", font_size=32).to_edge(UP, buff=0.5)
        
        self.prob_text = VGroup(
            Tex("Transmission: ", font_size=24).set_color(BLUE),
            Tex("Reflection: ", font_size=24).set_color(RED), 
            Tex("Barrier: ", font_size=24).set_color(GREEN),
            Tex("Total: ", font_size=24).set_color(WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UR, buff=0.3)
        
        self.prob_bg = SurroundingRectangle(
            self.prob_text, 
            color=WHITE, 
            fill_color=BLACK, 
            fill_opacity=0.8, 
            stroke_width=2,
            buff=0.15
        )
        
        self.time_display = Tex("t = 0.00", font_size=28).to_corner(UL, buff=0.3).shift(DOWN * 0.8 + RIGHT * 1.0)
        
        self.add(self.axes, x_label, y_label, self.barrier, barrier_label,
                title, self.prob_bg, self.prob_text, self.time_display)
        
    def animate_tunneling(self):
        max_prob = max(max(frame) for frame in self.prob_frames)
        scale = 0.7 / max_prob
        frame_skip = 3
        num_frames = len(self.prob_frames) // frame_skip
        
        initial_prob = np.array(self.prob_frames[0]) * scale
        points = [self.axes.coords_to_point(x, p) for x, p in zip(self.x_grid, initial_prob)]
        
        wave_curve = VMobject()
        wave_curve.set_points_smoothly(points)
        wave_curve.set_stroke(BLUE, width=4)
        self.add(wave_curve)
        
        for i in range(1, num_frames):
            frame_idx = i * frame_skip
            if frame_idx >= len(self.prob_frames):
                break
                
            current_prob = np.array(self.prob_frames[frame_idx]) * scale
            current_time = self.times[frame_idx]
            
            new_points = [self.axes.coords_to_point(x, p) for x, p in zip(self.x_grid, current_prob)]
            new_wave_curve = VMobject()
            new_wave_curve.set_points_smoothly(new_points)
            new_wave_curve.set_stroke(BLUE, width=4)
            
            trans_val = self.trans_prob[frame_idx] if frame_idx < len(self.trans_prob) else 0
            refl_val = self.refl_prob[frame_idx] if frame_idx < len(self.refl_prob) else 0
            barrier_val = self.barrier_prob[frame_idx] if frame_idx < len(self.barrier_prob) else 0
            # No absorbed value - removed absorber
            
            rounded_vals = self.simple_round([trans_val, refl_val, barrier_val])
            trans_rounded, refl_rounded, barrier_rounded = rounded_vals
            
            actual_sum = trans_val + refl_val + barrier_val
            total_rounded = round(actual_sum, 1)
            
            new_prob_text = VGroup(
                Tex(f"Transmission: {trans_rounded:.1f}", font_size=24).set_color(BLUE),
                Tex(f"Reflection: {refl_rounded:.1f}", font_size=24).set_color(RED),
                Tex(f"Barrier: {barrier_rounded:.1f}", font_size=24).set_color(GREEN),
                Tex(f"Total: {total_rounded:.1f}", font_size=24).set_color(WHITE)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UR, buff=0.3)
            
            new_prob_bg = SurroundingRectangle(
                new_prob_text, 
                color=WHITE, 
                fill_color=BLACK, 
                fill_opacity=0.8, 
                stroke_width=2,
                buff=0.15
            )
            
            new_time = Tex(f"t = {current_time:.2f}", font_size=28).to_corner(UL, buff=0.3).shift(DOWN * 0.8 + RIGHT * 1.0)
            
            self.play(
                Transform(wave_curve, new_wave_curve),
                Transform(self.prob_text, new_prob_text),
                Transform(self.prob_bg, new_prob_bg),
                Transform(self.time_display, new_time),
                run_time=0.15
            )
        
        self.wait(2)
    
    # def show_results(self):
    #     final_trans = self.trans_prob[-1]
    #     final_refl = self.refl_prob[-1] 
    #     final_barrier = self.barrier_prob[-1]
    #     final_absorbed = self.absorbed_prob[-1]
    #     final_total = final_trans + final_refl + final_barrier + final_absorbed
        
    #     results = VGroup(
    #         Tex("Final Results", font_size=32).set_color(YELLOW),
    #         Tex(f"Transmission: {final_trans:.1%}", font_size=26).set_color(BLUE),
    #         Tex(f"Reflection: {final_refl:.1%}", font_size=26).set_color(RED),
    #         Tex(f"Barrier: {final_barrier:.1%}", font_size=26).set_color(GREEN),
    #         Tex(f"Absorbed: {final_absorbed:.1%}", font_size=26).set_color(GRAY),
    #         Tex(f"Total: {final_total:.4f}", font_size=24).set_color(WHITE),
    #         Tex("Wave successfully absorbed\\\\at boundaries!", font_size=22).set_color(YELLOW)
    #     ).arrange(DOWN, buff=0.3).move_to(ORIGIN)
        
    #     bg = SurroundingRectangle(results, color=WHITE, fill_color=BLACK, fill_opacity=0.9, buff=0.5)
        
    #     self.play(FadeIn(bg), Write(results))
    #     self.wait(4)
    #     self.play(FadeOut(bg, results))
