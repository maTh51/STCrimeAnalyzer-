from experimental.sthsl.model import *
import torch
from experimental.sthsl.Params import args
from experimental.sthsl.DataHandler import DataHandler
from experimental.sthsl.engine import test
from experimental.sthsl.train import makePrint

def main():
    device = torch.device(args.device)
    handler = DataHandler(args.autotest)
    model = STHSL()
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print('model load successfully')

    with torch.no_grad():
        reses = test(model, handler)

    print(makePrint('Best', args.epoch, reses))
if __name__ == "__main__":
    main()
