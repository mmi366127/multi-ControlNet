
from ldm.models.diffusion.ddim import DDIMSampler
from utils import prompt_parser
import numpy as np

class VanillaStableDiffusionSampler(object):
    def __init__(self, constructor, model):
        self.sampler = constructor(model)
        self.is_plms = hasattr(self.sampler, 'p_sample_plms')
        self.orig_p_sample_ddim = self.sampler.p_sample_plms if self.is_plms else self.sampler.p_sample_ddim
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.sampler_noises = None
        self.step = 0
        self.stop_at = None
        self.eta = None
        self.config = None
        self.last_latent = None

    def p_sample_ddim_hook(self, x_dec, cond, ts, unconditional_conditioning, *args, **kwargs):

        image_conditioning = None
        if isinstance(cond, dict):
            image_conditioning = cond["c_concat"][0]
            cond = cond["c_crossattn"][0]
            unconditional_conditioning = unconditional_conditioning["c_crossattn"][0]
        
        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        unconditional_conditioning = prompt_parser.reconstruct_cond_batch(unconditional_conditioning, self.step)

        assert all([len(conds) == 1 for conds in conds_list]), 'composition via AND is not supported for DDIM/PLMS samplers'
        cond = tensor

        # for DDIM, shapes must match, we can't just process cond and uncond independently;
        # filling unconditional_conditioning with repeats of the last vector to match length is
        # not 100% correct but should work well enough
        if unconditional_conditioning.shape[1] < cond.shape[1]:
            last_vector = unconditional_conditioning[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - unconditional_conditioning.shape[1], 1])
            unconditional_conditioning = torch.hstack([unconditional_conditioning, last_vector_repeated])
        elif unconditional_conditioning.shape[1] > cond.shape[1]:
            unconditional_conditioning = unconditional_conditioning[:, :cond.shape[1]]

        if self.mask is not None:
            img_orig = self.sampler.model.q_sample(self.init_latent, ts)
            x_dec = img_orig * self.mask + self.nmask * x_dec

        # Wrap the image conditioning back up since the DDIM code can accept the dict directly.
        # Note that they need to be lists because it just concatenates them later.
        if image_conditioning is not None:
            cond = {"c_concat": [image_conditioning], "c_crossattn": [cond]}
            unconditional_conditioning = {"c_concat": [image_conditioning], "c_crossattn": [unconditional_conditioning]}

        res = self.orig_p_sample_ddim(x_dec, cond, ts, unconditional_conditioning=unconditional_conditioning, *args, **kwargs)

        if self.mask is not None:
            self.last_latent = self.init_latent * self.mask + self.nmask * res[1]
        else:
            self.last_latent = res[1]

        self.step += 1

        return res

    def init(self, eta=0.0, mask=None, nmask=None):

        self.eta = eta
        
        for fieldname in ['p_sample_ddim', 'p_sample_plms']:
            if hasattr(self.sampler, fieldname):
                setattr(self.sampler, fieldname, self.p_sample_ddim_hook)

        self.mask = mask
        self.nmask = nmask


    def sample(self, x, 
                     conditioning, 
                     unconditional_conditioning, 
                     unconditional_guidance_scale=7.0,
                     steps=20, 
                     image_conditioning=None, 
                     eta=0.0,
                     mask=None, 
                     nmask=None):

        self.init_latent = None
        self.last_latent = x
        self.step = 0

        self.init(eta=eta, mask=mask, nmask=nmask)

        if image_conditioning is not None:
            conditioning = {"dummy_for_plms": np.zeros((conditioning.shape[0],)), "c_crossattn": [conditioning], "c_concat": [image_conditioning]}
            unconditional_conditioning = {"c_crossattn": [unconditional_conditioning], "c_concat": [image_conditioning]}

        samples_ddim = self.sampler.sample(S=steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, x_T=x, eta=self.eta)[0]

        return samples_ddim

        