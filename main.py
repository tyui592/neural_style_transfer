"""Main Code."""
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from utils import imload, imsave
from loss import calc_content_loss, calc_gram_loss, calc_tv_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help="0 (noise) ~ 1 (content)")

    parser.add_argument('--content_image', type=str,
                        default="./imgs/golden_gate.jpg",
                        help="Content Image Path")

    parser.add_argument('--style_image', type=str,
                        default="./imgs/starry_night.jpg",
                        help="Style Image Path")

    parser.add_argument('--content_loss_weight', type=float, default=1.0,
                        help="Content Loss Weight")

    parser.add_argument('--style_loss_weight', type=float, default=50.0,
                        help="Style Loss Weight")

    parser.add_argument('--tv_loss_weight', type=float, default=0.5,
                        help="Total Variation Loss Weight")

    parser.add_argument('--iteration', type=int, default=1000,
                        help="Number of iterations")

    parser.add_argument('--imsize', type=int, default=256,
                        help="Image Size")

    parser.add_argument('--save_path', type=str, default='./imgs/',
                        help="Save dir path")
    args = parser.parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir()

    style_nodes = ['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1']
    content_nodes = ['relu_4_2']

    return_nodes = {'1': 'relu_1_1',
                    '6': 'relu_2_1',
                    '11': 'relu_3_1',
                    '20': 'relu_4_1',
                    '22': 'relu_4_2',
                    '29': 'relu_5_1'}

    device = torch.device('cuda')

    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
    vgg.eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    feature_extractor = create_feature_extractor(vgg,
                                                 return_nodes=return_nodes)

    content_image = imload(path=args.content_image,
                           imsize=args.imsize).to(device)
    style_image = imload(path=args.style_image,
                         imsize=args.imsize).to(device)

    # input image
    input_image = torch.rand_like(content_image).to(device)
    input_image = (input_image * (1 - args.noise_ratio)) \
        + (content_image.detach() * args.noise_ratio)

    # optimizer
    optimizer = torch.optim.LBFGS([input_image.requires_grad_(True)])

    for i in tqdm(range(args.iteration), desc='Stylization'):
        def closure():
            """Closure function."""
            optimizer.zero_grad()

            content_features = feature_extractor(content_image.detach())
            style_features = feature_extractor(style_image.detach())
            input_features = feature_extractor(input_image)

            content_loss = calc_content_loss(input_features,
                                             content_features,
                                             content_nodes)
            style_loss = calc_gram_loss(input_features,
                                        style_features,
                                        style_nodes)
            tv_loss = calc_tv_loss(input_image)

            total_loss = content_loss * args.content_loss_weight \
                + style_loss * args.style_loss_weight \
                + tv_loss * args.tv_loss_weight

            total_loss.backward()

            return total_loss

        # optimization
        optimizer.step(closure)

    # save image
    imsave(input_image, save_path / "stylized_image.png")
