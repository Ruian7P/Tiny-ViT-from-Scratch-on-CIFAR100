import torch
from model import VisionTransformer
from train import get_dataloaders, train_model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFAR100')
    
    # Model parameters
    parser.add_argument('--image_size', type=int, default=32, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--embed_dim', type=int, default=192, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=256, help='MLP dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)

    # Get dataloaders
    trainloader, testloader = get_dataloaders(args.image_size, args.batch_size)

    # Train model
    best_val_acc = train_model(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=args.num_epochs,
        device=device,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

if __name__ == "__main__":
    main() 