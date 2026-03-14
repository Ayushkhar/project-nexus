import torch
from nexus.inference.generate import generate_and_print_sample


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Computes cross entropy loss for a single batch.
    """

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Computes average loss over a dataloader.
    """

    total_loss = 0.

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):

        if i >= num_batches:
            break

        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluates model performance on train and validation sets.
    """

    model.eval()

    with torch.no_grad():

        train_loss = calc_loss_loader(
            train_loader,
            model,
            device,
            num_batches=eval_iter
        )

        val_loss = calc_loss_loader(
            val_loader,
            model,
            device,
            num_batches=eval_iter
        )

    model.train()

    return train_loss, val_loss


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer
):
    """
    Main training loop.
    """

    train_losses = []
    val_losses = []
    tokens_seen = []
    global_step = -1
    total_tokens = 0

    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device
            )

            loss.backward()
            optimizer.step()

            total_tokens += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen.append(total_tokens)

                print(
                    f"Epoch {epoch+1} | Step {global_step:06d} "
                    f"| Train Loss {train_loss:.3f} "
                    f"| Val Loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context
        )

    return train_losses, val_losses, tokens_seen