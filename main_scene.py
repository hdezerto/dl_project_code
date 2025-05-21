# main_scene.py
from manim import *
import os # Added for path checking, though try-except is primary
import xml.etree.ElementTree as ET # For explicitly catching ParseError if needed


class ReportAnimation(Scene):
    def construct(self):
        # Timings are approximate and should be adjusted
        # Total time: ~3 minutes 40 seconds

        self.show_cover()           
        self.show_intro()            # Approx 30s
        #self.show_baseline()         # Approx 40s
        #self.show_ssl()              # Approx 40s
        #self.show_vit()              # Approx 40s
        #self.show_lora()             # Approx 40s
        #self.show_conclusion()       # Approx 30s

        self.wait(5) # Hold final scene

    def create_title(self, text_content, scale=0.8, position=UP*3.5):
        title = Text(text_content, font_size=36, weight=BOLD).scale(scale).move_to(position)
        underline = Underline(title, color=BLUE)
        return VGroup(title, underline)
    

    def show_cover(self):
        # Project Title (centered)
        title = Text("Exploring Transfer Learning", font_size=50, weight=BOLD, color=BLUE_D)
        title.move_to(ORIGIN)

        # KTH Logo (top right, small, fade in)
        logo_path = os.path.join("assets", "kth_logo.png")
        logo = None
        if os.path.exists(logo_path):
            logo = ImageMobject(logo_path).scale(0.15)
            logo.to_corner(UR, buff=0.5)

        # Institution phrase (centered, below title)
        institution = Text("Royal Institute of Technology (KTH), Sweden", font_size=24, color=GRAY_D)
        institution.next_to(title, DOWN, buff=0.6)

        # Group Members (centered, near bottom, smaller font)
        members = VGroup(
            Text("Diogo Paulo", font_size=20),
            Text("Hugo Dezerto", font_size=20),
            Text("Maria Carolina Sebastião", font_size=20)
        ).arrange(DOWN, buff=0.15)
        members.move_to(DOWN * 3)

        # Animate
        self.play(Write(title))
        if logo:
            self.play(FadeIn(logo))
        self.wait(0.3)
        self.play(Write(institution))
        self.wait(0.3)
        self.play(LaggedStart(*[Write(m) for m in members], lag_ratio=0.3))
        self.wait(2)

        # Fade out all cover elements before next scene
        fadeout_objs = [title, institution, members]
        if logo:
            fadeout_objs.append(logo)
        self.play(*[FadeOut(obj) for obj in fadeout_objs])


    def show_intro(self):
        # Title
        title = self.create_title("Introduction: Exploring Transfer Learning")
        self.play(Write(title))
        self.wait(0.5)
    
        # Definition
        definition = Text(
            "Transfer learning uses knowledge from one task to solve another.",
            font_size=28, t2c={"knowledge": YELLOW, "another": BLUE}
        ).next_to(title, DOWN, buff=0.7)
        self.play(Write(definition))
        self.wait(1.5)
    
        # Visual: Task A -> Knowledge -> Task B
        task_a = Text("Task A\n(e.g., ImageNet)", font_size=24).to_edge(LEFT, buff=1).shift(UP*1)
        task_b = Text("Task B\n(Pet Classification)", font_size=24).to_edge(RIGHT, buff=1).shift(UP*1)
        knowledge = Rectangle(width=2, height=1, color=GREEN, fill_opacity=0.5).move_to(ORIGIN)
        knowledge_text = Text("Knowledge", font_size=18).move_to(knowledge.get_center())
        knowledge_vg = VGroup(knowledge, knowledge_text)
        arrow = Arrow(task_a.get_right(), knowledge.get_left(), buff=0.1, color=YELLOW)
        arrow2 = Arrow(knowledge.get_right(), task_b.get_left(), buff=0.1, color=YELLOW)
    
        self.play(FadeIn(task_a), FadeIn(task_b), FadeIn(knowledge_vg))
        self.play(GrowArrow(arrow), GrowArrow(arrow2))
        self.wait(1)
    
        # Motivation
        motivation = Text(
            "Especially useful when data for the target task is limited.",
            font_size=24, color=GRAY_D
        ).next_to(definition, DOWN, buff=0.5)
        self.play(Write(motivation))
        self.wait(1.5)
    
        # Fade out visuals before aims
        self.play(
            FadeOut(task_a), FadeOut(task_b), FadeOut(knowledge_vg),
            FadeOut(arrow), FadeOut(arrow2), FadeOut(definition), FadeOut(motivation)
        )
    
        # Project Aims
        aims_title = Text("Project Aims:", font_size=28).next_to(title, DOWN, buff=1)
        aim1 = Text("1. Improve ResNet Baseline", font_size=24).next_to(aims_title, DOWN, buff=0.3, aligned_edge=LEFT)
        aim2 = Text("2. Semi-Supervised Learning (SSL)", font_size=24).next_to(aim1, DOWN, buff=0.2, aligned_edge=LEFT)
        aim3 = Text("3. Vision Transformers (ViT)", font_size=24).next_to(aim2, DOWN, buff=0.2, aligned_edge=LEFT)
        aim4 = Text("4. LoRA Layers", font_size=24).next_to(aim3, DOWN, buff=0.2, aligned_edge=LEFT)
    
        self.play(Write(aims_title))
        self.play(LaggedStart(Write(aim1), Write(aim2), Write(aim3), Write(aim4), lag_ratio=0.5))
        self.wait(2)



    def show_intro(self):
        # Title
        title = self.create_title("Transfer Learning")
        self.play(Write(title))
        self.wait(0.3)

        # Definition (keep on screen)
        definition = Text(
            "Transfer learning is a deep learning technique where knowledge gained\n"
            "from solving one task is reused to solve a different, but related, task.",
            font_size=22, t2c={"knowledge": YELLOW, "different": BLUE}
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(definition))
        self.wait(0.5)

        # Visual: Task A -> Knowledge -> Task B
        task_a = Rectangle(width=2, height=1, color=BLUE_D, fill_opacity=0.2)
        task_a_text = Text("Task A\n(ImageNet)", font_size=22).move_to(task_a.get_center())
        task_a_group = VGroup(task_a, task_a_text)
        
        task_b = Rectangle(width=4, height=1, color=GREEN_D, fill_opacity=0.2)
        task_b_text = Text("Task B\n(Pet Classification)", font_size=22).move_to(task_b.get_center())
        task_b_group = VGroup(task_b, task_b_text)
        
        knowledge = Rectangle(width=2, height=1, color=YELLOW, fill_opacity=0.4)
        knowledge_text = Text("Knowledge", font_size=20, color=BLACK).move_to(knowledge.get_center())
        knowledge_group = VGroup(knowledge, knowledge_text)
        
        # Arrange horizontally and place below definition
        visual_group = VGroup(task_a_group, knowledge_group, task_b_group).arrange(RIGHT, buff=1.2)
        visual_group.next_to(definition, DOWN, buff=0.7)
        
        arrow1 = Arrow(task_a_group.get_right(), knowledge_group.get_left(), buff=0.1, color=YELLOW)
        arrow2 = Arrow(knowledge_group.get_right(), task_b_group.get_left(), buff=0.1, color=YELLOW)
        
        self.play(FadeIn(task_a_group), FadeIn(task_b_group), FadeIn(knowledge_group))
        self.play(GrowArrow(arrow1), GrowArrow(arrow2))
        self.wait(0.5)
        self.play(Indicate(knowledge_group))
        self.wait(0.5)
        
        # Your case (keep on screen)
        your_case = Text(
            "We fine-tuned a ResNet18 model pre-trained on ImageNet\n"
            "using the Oxford-IIIT Pet Dataset (37 breeds of cats and dogs).",
            font_size=22
        ).next_to(knowledge_group, DOWN, buff=0.4)
        self.play(Write(your_case))
        self.wait(0.5)

        # Baseline and aims (keep on screen)
        baseline = Text(
            "Compared against:",
            font_size=22
        ).next_to(your_case, DOWN, buff=0.3)
        self.play(Write(baseline))
        self.wait(0.3)

        aim1 = Text("1. Semi-supervised learning with pseudo-labeling", font_size=20, color=BLUE_D).next_to(baseline, DOWN, buff=0.2, aligned_edge=LEFT)
        aim2 = Text("2. Vision Transformers (ViT)", font_size=20, color=ORANGE).next_to(aim1, DOWN, buff=0.1, aligned_edge=LEFT)
        aim3 = Text("3. LoRA layers for parameter-efficient fine-tuning", font_size=20, color=RED_D).next_to(aim2, DOWN, buff=0.1, aligned_edge=LEFT)
        self.play(LaggedStart(Write(aim1), Write(aim2), Write(aim3), lag_ratio=0.4))
        
        self.wait(2)






    def show_baseline(self):
        title = self.create_title("Baseline: ResNet18")
        self.play(Write(title))
        self.wait(1)

        # Visual: Simplified CNN (ResNet18)
        input_layer = Rectangle(width=1, height=2, color=BLUE, fill_opacity=0.3).set_z_index(0)
        input_text = Text("Input Image", font_size=18).move_to(input_layer.get_center())
        input_group = VGroup(input_layer, input_text).shift(LEFT*5 + UP*1)

        resnet_blocks = VGroup()
        block_colors = [RED_E, RED_D, RED_C, RED_B] # Representing layers
        for i in range(4):
            block = Rectangle(width=1, height=2 - i*0.2, color=block_colors[i], fill_opacity=0.5).set_z_index(1)
            resnet_blocks.add(block)
        resnet_blocks.arrange(RIGHT, buff=0.2).next_to(input_layer, RIGHT, buff=0.2)
        resnet_text = Text("ResNet18 Backbone", font_size=20).next_to(resnet_blocks, DOWN, buff=0.2)

        fc_layer_old = Rectangle(width=1, height=1, color=GRAY, fill_opacity=0.5).next_to(resnet_blocks, RIGHT, buff=0.2)
        fc_text_old = Text("Old FC\n(1000 cls)", font_size=14).move_to(fc_layer_old.get_center()) # Shorter text
        fc_group_old = VGroup(fc_layer_old, fc_text_old)

        self.play(FadeIn(input_group), FadeIn(resnet_blocks), FadeIn(resnet_text), FadeIn(fc_group_old))
        self.wait(1.5)

        # Replacing FC layer
        fc_layer_new_binary = Rectangle(width=1, height=0.5, color=GREEN_C, fill_opacity=0.5).align_to(fc_layer_old, LEFT).shift(UP*0.5)
        fc_text_new_binary = Text("New FC\n(2 cls)", font_size=14).move_to(fc_layer_new_binary.get_center())
        fc_group_new_binary = VGroup(fc_layer_new_binary, fc_text_new_binary)

        fc_layer_new_multi = Rectangle(width=1, height=0.7, color=GREEN_E, fill_opacity=0.5).align_to(fc_layer_old, LEFT).shift(DOWN*0.5)
        fc_text_new_multi = Text("New FC\n(37 cls)", font_size=14).move_to(fc_layer_new_multi.get_center())
        fc_group_new_multi = VGroup(fc_layer_new_multi, fc_text_new_multi)
        
        # Store the original position for fc_group_old to revert to if needed, or use .copy()
        original_fc_pos = fc_group_old.get_center()

        self.play(Transform(fc_group_old, fc_group_new_binary.copy().move_to(original_fc_pos)))
        self.wait(1)
        # For the second transform, ensure fc_group_old is reset or use a new target
        # It's often cleaner to transform a copy or manage state carefully.
        # Here, fc_group_old is already transformed. We want to transform it to the multi-class version.
        self.play(Transform(fc_group_old, fc_group_new_multi.copy().move_to(original_fc_pos)))
        self.wait(1.5)

        # Fine-tuning strategy (Simultaneous)
        fine_tuning_text = Text("Fine-Tuning Strategy 1 (Simultaneous):", font_size=24).next_to(title, DOWN, buff=0.7).to_edge(LEFT, buff=0.5)
        unfreeze_text = Text("Unfreeze last residual blocks + FC layer", font_size=20).next_to(fine_tuning_text, DOWN, buff=0.3, aligned_edge=LEFT)
        
        highlight_boxes = VGroup()
        if len(resnet_blocks) >= 4: # Ensure there are enough blocks
            for i in range(max(0, len(resnet_blocks)-2), len(resnet_blocks)): # Highlight last 2 blocks safely
                highlight_boxes.add(SurroundingRectangle(resnet_blocks[i], color=YELLOW, buff=0.05, stroke_width=2))
        highlight_boxes.add(SurroundingRectangle(fc_group_old, color=YELLOW, buff=0.05, stroke_width=2)) # fc_group_old is now the multi-class one

        self.play(Write(fine_tuning_text))
        self.play(Write(unfreeze_text), Create(highlight_boxes))
        self.wait(2)

        # Results
        results_text_binary = Text("Binary Classification Accuracy: 99.02%", font_size=24, t2c={"99.02%": GREEN}).next_to(unfreeze_text, DOWN, buff=0.7, aligned_edge=LEFT)
        results_text_multi = Text("Multi-class Accuracy (Strategy 1): 90.54%", font_size=24, t2c={"90.54%": GREEN}).next_to(results_text_binary, DOWN, buff=0.3, aligned_edge=LEFT)
        
        self.play(Write(results_text_binary))
        self.wait(1)
        self.play(Write(results_text_multi))
        self.wait(3)

        self.play(FadeOut(title), FadeOut(input_group), FadeOut(resnet_blocks), FadeOut(resnet_text), FadeOut(fc_group_old), FadeOut(fine_tuning_text), FadeOut(unfreeze_text), FadeOut(highlight_boxes), FadeOut(results_text_binary), FadeOut(results_text_multi))


    def show_ssl(self):
        title = self.create_title("Semi-Supervised Learning (SSL)")
        self.play(Write(title))
        self.wait(1)

        challenge_text = Text("Challenge: Limited Labeled Data", font_size=28).next_to(title, DOWN, buff=0.5)
        self.play(Write(challenge_text))
        self.wait(1)

        # Visual: Small pile of labeled data vs large pile of unlabeled data
        # Try to load SVGs, with fallback
        labeled_icon_path = os.path.join("assets", "labeled_icon.svg")
        unlabeled_icon_path = os.path.join("assets", "unlabeled_icon.svg")

        try:
            # Check if file exists and is not empty before attempting to load as SVGMobject
            if os.path.exists(labeled_icon_path) and os.path.getsize(labeled_icon_path) > 0:
                labeled_icon_svg = SVGMobject(labeled_icon_path)
                if labeled_icon_svg.get_num_submobjects() == 0 and len(labeled_icon_svg.points) == 0:
                    raise ValueError("Empty or invalid SVG structure for labeled_icon.svg")
                labeled_icon = labeled_icon_svg.scale(0.3)
            else:
                raise FileNotFoundError("labeled_icon.svg not found or is empty.")
        except Exception as e:
            print(f"INFO: Could not load or parse '{labeled_icon_path}' ({type(e).__name__}: {e}). Using fallback.")
            labeled_shape = Circle(radius=0.3, color=BLUE, fill_opacity=1).set_z_index(1) # Slightly larger for visibility
            labeled_text_obj = Text("L", font_size=24, color=WHITE, weight=BOLD).move_to(labeled_shape.get_center()).set_z_index(2)
            labeled_icon = VGroup(labeled_shape, labeled_text_obj).scale(0.7) # Scale the group

        try:
            if os.path.exists(unlabeled_icon_path) and os.path.getsize(unlabeled_icon_path) > 0:
                unlabeled_icon_svg = SVGMobject(unlabeled_icon_path)
                if unlabeled_icon_svg.get_num_submobjects() == 0 and len(unlabeled_icon_svg.points) == 0:
                    raise ValueError("Empty or invalid SVG structure for unlabeled_icon.svg")
                unlabeled_icon = unlabeled_icon_svg.scale(0.3)
            else:
                raise FileNotFoundError("unlabeled_icon.svg not found or is empty.")
        except Exception as e:
            print(f"INFO: Could not load or parse '{unlabeled_icon_path}' ({type(e).__name__}: {e}). Using fallback.")
            unlabeled_shape = Circle(radius=0.3, color=GRAY, fill_opacity=1).set_z_index(1) # Slightly larger
            unlabeled_text_obj = Text("U", font_size=24, color=BLACK, weight=BOLD).move_to(unlabeled_shape.get_center()).set_z_index(2)
            unlabeled_icon = VGroup(unlabeled_shape, unlabeled_text_obj).scale(0.7) # Scale the group
        
        labeled_pile = VGroup(*[labeled_icon.copy().shift(RIGHT*i*0.1 + DOWN*j*0.1) for i in range(3) for j in range(2)]).scale(0.8) # Adjusted scale
        unlabeled_pile = VGroup(*[unlabeled_icon.copy().shift(RIGHT*i*0.1 + DOWN*j*0.1) for i in range(8) for j in range(4)]).scale(0.8) # Adjusted scale
        
        labeled_pile.next_to(challenge_text, DOWN, buff=0.5).to_edge(LEFT, buff=2)
        unlabeled_pile.next_to(labeled_pile, RIGHT, buff=1)
        
        labeled_text_desc = Text("Labeled Data (Small)", font_size=20).next_to(labeled_pile, DOWN, buff=0.1)
        unlabeled_text_desc = Text("Unlabeled Data (Large)", font_size=20).next_to(unlabeled_pile, DOWN, buff=0.1)

        self.play(FadeIn(labeled_pile), FadeIn(labeled_text_desc))
        self.play(FadeIn(unlabeled_pile), FadeIn(unlabeled_text_desc))
        self.wait(2)

        # Pseudo-labeling process
        process_title = Text("Pseudo-Labeling Process:", font_size=24).next_to(labeled_pile, RIGHT, buff=2).align_to(challenge_text, LEFT).shift(RIGHT*4)
        step1 = Text("1. Train model on Labeled Data", font_size=20).next_to(process_title, DOWN, buff=0.3, aligned_edge=LEFT)
        step2 = Text("2. Predict on Unlabeled Data (get Pseudo-Labels)", font_size=20).next_to(step1, DOWN, buff=0.2, aligned_edge=LEFT)
        step3 = Text("3. Combine Labeled + Pseudo-Labeled Data", font_size=20).next_to(step2, DOWN, buff=0.2, aligned_edge=LEFT)
        step4 = Text("4. Retrain model on combined set", font_size=20).next_to(step3, DOWN, buff=0.2, aligned_edge=LEFT)
        
        self.play(Write(process_title))
        self.play(Write(step1))
        model_mockup = Rectangle(width=1, height=0.7, color=PURPLE).next_to(labeled_pile, RIGHT, buff=0.2)
        self.play(FadeIn(model_mockup), labeled_pile.animate.shift(LEFT*0.2))
        self.play(labeled_pile.animate.shift(RIGHT*0.2)) 
        self.wait(1)

        self.play(Write(step2))
        pseudo_labels_on_unlabeled_group = VGroup()
        # Create a temporary copy of unlabeled_pile to show transformation, then fade original
        unlabeled_pile_copy_for_pseudo = unlabeled_pile.copy()
        for item_idx, item in enumerate(unlabeled_pile_copy_for_pseudo):
            pseudo_label = Text("P", font_size=12, color=GREEN).move_to(item.get_center()+UP*0.1+RIGHT*0.1) # Changed GREEN_SCREEN to GREEN
            pseudo_labels_on_unlabeled_group.add(pseudo_label)
        
        self.play(Transform(unlabeled_pile_copy_for_pseudo, pseudo_labels_on_unlabeled_group))
        self.wait(1)

        self.play(Write(step3))
        self.play(Write(step4))
        combined_data_pos = model_mockup.get_center() + DOWN*1.5
        self.play(labeled_pile.animate.scale(0.7).move_to(combined_data_pos + LEFT*1),
                  unlabeled_pile_copy_for_pseudo.animate.scale(0.7).move_to(combined_data_pos + RIGHT*1)) # Use the transformed pile
        self.play(model_mockup.animate.move_to(combined_data_pos + UP*1.5)) 
        self.wait(2)

        # Results: Bar chart
        ssl_results_text = Text("Results: Significant gains with few labels", font_size=24).move_to(DOWN*2.5)
        
        chart_data_supervised = [42.9, 77.8] 
        chart_data_ssl = [51.5, 81.4]       
        
        chart = BarChart(
            values=[chart_data_supervised[0], chart_data_ssl[0], chart_data_supervised[1], chart_data_ssl[1]],
            bar_names=["Sup.", "SSL", "Sup.", "SSL"], # Shorter names
            y_range=[0, 100, 20],
            y_length=3,
            x_length=5,
            bar_colors=[BLUE_D, GREEN_D, BLUE_D, GREEN_D], # Consistent colors for types
            x_axis_config={"font_size": 16},
            y_axis_config={"font_size": 16},
            # bar_label_scale=0.6,   # <-- REMOVED
        ).scale(0.7).next_to(ssl_results_text, UP, buff=0.5)
        
        # Add labels for groups of bars
        group_label_1_percent = Text("1% Labeled", font_size=18).next_to(chart.bars[0:2], DOWN, buff=0.3)
        group_label_10_percent = Text("10% Labeled", font_size=18).next_to(chart.bars[2:4], DOWN, buff=0.3)

        self.play(FadeOut(labeled_pile), FadeOut(unlabeled_pile), FadeOut(labeled_text_desc), FadeOut(unlabeled_text_desc), 
                  FadeOut(model_mockup), FadeOut(process_title), FadeOut(step1), FadeOut(step2), FadeOut(step3), FadeOut(step4),
                  FadeOut(unlabeled_pile_copy_for_pseudo)) # Fade out the transformed copy
        self.play(Write(ssl_results_text))
        self.play(Create(chart), Write(group_label_1_percent), Write(group_label_10_percent))
        self.wait(4)

        self.play(FadeOut(title), FadeOut(challenge_text), FadeOut(ssl_results_text), FadeOut(chart), FadeOut(group_label_1_percent), FadeOut(group_label_10_percent))


    def show_vit(self):
        title = self.create_title("Vision Transformers (ViT)")
        self.play(Write(title))
        self.wait(1)

        vit_intro = Text("ViT-Base-Patch16-224: Divides image into patches", font_size=24).next_to(title, DOWN, buff=0.7)
        self.play(Write(vit_intro))

        image_placeholder = Square(side_length=2, color=LIGHT_GRAY, fill_opacity=0.3).next_to(vit_intro, DOWN, buff=0.5).shift(LEFT*3)
        image_text_label = Text("Image", font_size=18).move_to(image_placeholder) # Renamed to avoid conflict
        self.play(FadeIn(image_placeholder), Write(image_text_label))

        patches_group = VGroup() # Renamed to avoid conflict
        patch_side_length = image_placeholder.side_length / 4 
        for i in range(4):
            for j in range(4):
                patch = Square(side_length=patch_side_length, color=BLUE_C, fill_opacity=0.5, stroke_width=1, stroke_color=BLUE_E)
                patch.move_to(image_placeholder.get_corner(UL) + RIGHT*(j*patch_side_length + patch_side_length/2) + DOWN*(i*patch_side_length + patch_side_length/2))
                patches_group.add(patch)
        
        # Animate image breaking into patches
        # Create copies of patches from the image_placeholder's current state for the transform
        image_placeholder_copy = image_placeholder.copy()
        self.play(LaggedStart(*[Transform(image_placeholder_copy.copy(), p.copy()) for p in patches_group], lag_ratio=0.05, run_time=2))
        # Fade out the original placeholder after patches appear from it
        self.play(FadeOut(image_placeholder_copy), FadeIn(patches_group)) # Fade in the actual group of patches
        self.wait(1)


        sequence_text = Text("Patches -> Sequence -> Transformer Encoders", font_size=20).next_to(patches_group, RIGHT, buff=1).align_to(patches_group, UP)
        self.play(Write(sequence_text))

        patches_seq_display = patches_group.copy().arrange(RIGHT, buff=0.1).scale(0.3).next_to(sequence_text, DOWN, buff=0.2)
        transformer_block = Rectangle(width=3, height=1, color=ORANGE, fill_opacity=0.5).next_to(patches_seq_display, DOWN, buff=0.2)
        transformer_text_label = Text("Transformer Encoder\n(Self-Attention)", font_size=16).move_to(transformer_block) # Renamed
        
        self.play(Transform(patches_group.copy(), patches_seq_display)) # Transform a copy
        self.play(FadeIn(transformer_block), Write(transformer_text_label))
        self.wait(2)

        advantage_text = Text(
            "Advantage: Captures long-range dependencies & global context",
            font_size=22,
            t2c={"global context": YELLOW}
        ).next_to(transformer_block, DOWN, buff=0.7)
        self.play(Write(advantage_text))
        self.wait(2)

        vit_result = Text("ViT Test Accuracy: 92.34%", font_size=28, t2c={"92.34%": GREEN_C}).next_to(advantage_text, DOWN, buff=0.7)
        baseline_comparison = Text("(Baseline ResNet18: 90.54%)", font_size=20).next_to(vit_result, DOWN, buff=0.1)
        self.play(Write(vit_result), Write(baseline_comparison))
        self.wait(3)

        # Collect all mobjects created in this scene for FadeOut
        # Note: Some mobjects are transformed (e.g. fc_group_old). Ensure all relevant versions are handled or use .copy() for transforms.
        # For simplicity, fading out explicitly named mobjects.
        # The patches_group was used for the sequence display, so it's implicitly handled if patches_seq_display is a transform of its copy.
        self.play(FadeOut(title), FadeOut(vit_intro), FadeOut(image_text_label), 
                  FadeOut(patches_group), # This is the group of 16 patches on the left
                  FadeOut(sequence_text), FadeOut(patches_seq_display), # This is the sequence on the right
                  FadeOut(transformer_block), FadeOut(transformer_text_label), 
                  FadeOut(advantage_text), FadeOut(vit_result), FadeOut(baseline_comparison),
                  FadeOut(image_placeholder)) # Ensure original placeholder is also faded if it wasn't fully transformed away


    def show_lora(self):
        title = self.create_title("LoRA: Low-Rank Adaptation")
        self.play(Write(title))
        self.wait(1)

        intro_text = Text("Efficient fine-tuning of pre-trained models", font_size=24).next_to(title, DOWN, buff=0.5)
        self.play(Write(intro_text))
        self.wait(1)

        lora_equation = MathTex(
            r"W' = W + \frac{\alpha}{r} B A", 
            font_size=40
        ).next_to(intro_text, DOWN, buff=0.7)
        
        w_desc = Tex("$W$: Pre-trained (Frozen)", font_size=28).next_to(lora_equation, DOWN, buff=0.5, aligned_edge=LEFT) # Renamed
        ba_desc = Tex("$B, A$: Trainable Low-Rank Matrices", font_size=28).next_to(w_desc, DOWN, buff=0.2, aligned_edge=LEFT) # Renamed
        r_desc = Tex("$r$: Rank (small, e.g., 4 or 8)", font_size=28).next_to(ba_desc, DOWN, buff=0.2, aligned_edge=LEFT) # Renamed

        self.play(Write(lora_equation))
        self.wait(1)
        self.play(Write(w_desc))
        self.play(Write(ba_desc))
        self.play(Write(r_desc))
        self.wait(3)

        matrix_w = Square(side_length=2, color=BLUE_E, fill_opacity=0.3).next_to(lora_equation, RIGHT, buff=1.5).shift(UP*0.5)
        matrix_w_label = Text("W (Large, Frozen)", font_size=18).next_to(matrix_w, UP, buff=0.1)
        
        matrix_b = Rectangle(width=0.5, height=2, color=RED_E, fill_opacity=0.5).next_to(matrix_w, RIGHT, buff=0.5, aligned_edge=UP)
        matrix_a = Rectangle(width=2, height=0.5, color=RED_E, fill_opacity=0.5).next_to(matrix_b, DOWN, buff=0.1, aligned_edge=LEFT)
        ba_label = Text("B  A (Small, Trainable Update)", font_size=18).next_to(VGroup(matrix_b,matrix_a), UP, buff=0.1)
        plus_sign = MathTex("+").scale(2).move_to(midpoint(matrix_w.get_right(), matrix_b.get_left()))

        self.play(FadeIn(matrix_w), FadeIn(matrix_w_label))
        self.play(FadeIn(matrix_b), FadeIn(matrix_a), FadeIn(ba_label), Write(plus_sign))
        self.wait(2)
        self.play(FadeOut(w_desc), FadeOut(ba_desc), FadeOut(r_desc)) 

        efficiency_title = Text("Parameter Efficiency (ResNet18 + LoRA)", font_size=24).move_to(lora_equation.get_center()).shift(DOWN*1.0)
        params_lora = Text("LoRA: ~71,680 trainable params (r=4)", font_size=22, t2c={"~71,680": GREEN_D}).next_to(efficiency_title, DOWN, buff=0.3)
        params_baseline = Text("Baseline Fine-tune: ~11,000,000 trainable params", font_size=22, t2c={"~11,000,000": RED_D}).next_to(params_lora, DOWN, buff=0.2)
        percentage_text = Text("~0.65% of baseline parameters!", font_size=24, weight=BOLD, color=YELLOW_D).next_to(params_baseline, DOWN, buff=0.3)

        self.play(Write(efficiency_title))
        self.play(Write(params_lora))
        self.play(Write(params_baseline))
        self.play(Write(percentage_text))
        self.wait(3)

        accuracy_text = Text("Achieves competitive accuracy (e.g., 89.26% with ResNet18 LoRA)", font_size=22).next_to(percentage_text, DOWN, buff=0.5)
        self.play(Write(accuracy_text))
        self.wait(3)
        
        self.play(FadeOut(title), FadeOut(intro_text), FadeOut(lora_equation), FadeOut(matrix_w), FadeOut(matrix_w_label), FadeOut(matrix_b), FadeOut(matrix_a), FadeOut(ba_label), FadeOut(plus_sign), FadeOut(efficiency_title), FadeOut(params_lora), FadeOut(params_baseline), FadeOut(percentage_text), FadeOut(accuracy_text))


    def show_conclusion(self):
        title = self.create_title("Conclusion & Key Takeaways")
        self.play(Write(title))
        self.wait(1)

        takeaway1 = Text("• Strong ResNet18 baseline (>90% accuracy).", font_size=24).next_to(title, DOWN, buff=0.7, aligned_edge=LEFT).shift(LEFT*3) # Shifted more left
        takeaway2 = Text("• SSL (Pseudo-labeling) boosts performance with scarce labels\n  (+9% for 1% data).", t2c={"SSL":BLUE, "Pseudo-labeling":BLUE},font_size=24).next_to(takeaway1, DOWN, buff=0.3, aligned_edge=LEFT)
        takeaway3 = Text("• Vision Transformers surpassed CNNs (92.34% accuracy).", t2c={"Vision Transformers":ORANGE},font_size=24).next_to(takeaway2, DOWN, buff=0.3, aligned_edge=LEFT)
        takeaway4 = Text("• LoRA enables efficient fine-tuning (<1% params, high accuracy).", t2c={"LoRA":RED},font_size=24).next_to(takeaway3, DOWN, buff=0.3, aligned_edge=LEFT)
        
        takeaways = VGroup(takeaway1, takeaway2, takeaway3, takeaway4).scale_to_fit_width(self.camera.frame_width - 1) # Ensure it fits
        takeaways.move_to(ORIGIN + DOWN*0.5) # Center it a bit lower
        self.play(LaggedStart(*[Write(t) for t in takeaways], lag_ratio=0.7))
        self.wait(4)

        future_work_title = Text("Future Work:", font_size=28, weight=BOLD).next_to(takeaways, DOWN, buff=0.7, aligned_edge=LEFT)
        future_work_text = Text("Combine LoRA with Vision Transformers for further efficiency and accuracy.", font_size=24).next_to(future_work_title, DOWN, buff=0.3, aligned_edge=LEFT)
        
        future_work_group = VGroup(future_work_title, future_work_text).next_to(takeaways, DOWN, buff=0.5).align_to(takeaways, LEFT)


        vit_block_future = Rectangle(width=1.5, height=1, color=ORANGE, fill_opacity=0.5).next_to(future_work_group, DOWN, buff=0.5).shift(LEFT*1)
        vit_text_future = Text("ViT", font_size=18).move_to(vit_block_future)
        lora_module_future = Rectangle(width=0.5, height=0.7, color=RED, fill_opacity=0.5).next_to(vit_block_future, RIGHT, buff=0.1, aligned_edge=UP)
        lora_text_future = Text("LoRA", font_size=12).move_to(lora_module_future)
        plus_sign_future = MathTex("+").scale(1.5).move_to(midpoint(vit_block_future.get_right(), lora_module_future.get_left()))
        future_combo = VGroup(vit_block_future, vit_text_future, lora_module_future, lora_text_future, plus_sign_future)
        
        arrow_up = Arrow(future_combo.get_bottom() + DOWN*0.2, future_combo.get_top() + UP*0.2, color=GREEN_D, buff=0.1) # Adjusted arrow
        accuracy_up_text = Text("Accuracy ↑", font_size=18, color=GREEN_D).next_to(arrow_up.get_end(), RIGHT, buff=0.1)
        params_down_text = Text("Params ↓", font_size=18, color=GREEN_D).next_to(arrow_up.get_end(), LEFT, buff=0.1)
        future_visuals = VGroup(future_combo, arrow_up, accuracy_up_text, params_down_text)

        self.play(Write(future_work_group))
        self.play(FadeIn(future_visuals))
        self.wait(4)

        final_text = Text("Thank You!", font_size=36, color=BLUE_C).move_to(DOWN*3.5) # Adjusted position
        self.play(Write(final_text))
        self.wait(2) # Hold thank you message a bit
        
        # Fade out all elements from conclusion for a clean end
        self.play(FadeOut(title), FadeOut(takeaways), FadeOut(future_work_group), FadeOut(future_visuals), FadeOut(final_text))
