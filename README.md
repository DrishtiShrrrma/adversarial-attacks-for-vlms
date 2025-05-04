# Pseudocode for PGD attack
for step in range(num_steps):
    adv_image = original_image + perturbation
    loss = -model(image=adv_image).logits.mean()
    perturbation += step_size * normalized_gradient(loss)
    perturbation = project_to_valid_range(perturbation, norm, epsilon)
