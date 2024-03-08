from models.coil import COIL
from models.der import DER
from models.der_new import DER as DER_new
from models.ewc import EWC
from models.finetune import Finetune
from models.finetune_new import Finetune as Finetune_new
from models.gem import GEM
from models.icarl import iCaRL
from models.icarl_new import iCaRL_new
from models.lwf import LwF
from models.lwf_new import LwF_new
from models.e_lwf import E_LwF
from models.replay import Replay
from models.replay_new import Replay as Replay_new
from models.bic import BiC
from models.bic_new import BiC as BiC_new
from models.podnet import PODNet
from models.podnet_new import PODNet as PODNet_new
from models.wa import WA
from models.wa_new import WA_new
from models.gms import GMS
from models.e_finetune import E_Finetune
from models.simclr import SimCLR
from models.ssl_ce import Ssl_ce
from models.ssl_ce_1 import Ssl_ce_1
from models.my import My
from models.meta_attention import Meta_attention

def get_model(model_name, args):
    name = model_name.lower()
    if name == 'icarl':
        return iCaRL(args)
    if name == 'icarl_new':
        return iCaRL_new(args)
    elif name == 'bic':
        return BiC(args)
    elif name == 'bic_new':
        return BiC_new(args)
    elif name == 'podnet':
        return PODNet(args)
    elif name == 'podnet_new':
        return PODNet_new(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "lwf_new":
        return LwF_new(args)
    elif name == "e_lwf":
        return E_LwF(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "wa":
        return WA(args)
    elif name == "wa_new":
        return WA_new(args)
    elif name == "der":
        return DER(args)
    elif name == "der_new":
        return DER_new(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "finetune_new":
        return Finetune_new(args)
    elif name == "e_finetune":
        return E_Finetune(args)
    elif name == "replay":
        return Replay(args)
    elif name == "replay_new":
        return Replay_new(args)
    elif name == "gem":
        return GEM(args)
    elif name == "gms":
        return GMS(args)
    elif name == "coil":
        return COIL(args)
    elif name == "simclr":
        return SimCLR(args)
    elif name == "ssl_ce":
        return Ssl_ce(args)
    elif name == "ssl_ce_1":
        return Ssl_ce_1(args)
    elif name == "my":
        return My(args)
    elif name == "meta_attention":
        return Meta_attention(args)
    else:
        assert 0
