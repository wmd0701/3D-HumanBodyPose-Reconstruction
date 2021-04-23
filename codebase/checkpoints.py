import os
import urllib
import torch
from torch.utils import model_zoo


class CheckpointIO(object):
    """ CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    """

    def __init__(self, checkpoint_dir='./chkpts', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        self.vis_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        """ Registers modules in current module dictionary.
        """
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        """ Saves the current module dictionary.

        Args:
            filename (str): name of output file
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename):
        """ Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        """
        if urllib.parse.urlparse(filename).scheme in ('http', 'https'):
            return self.load_url(filename)
        else:
            return self.load_file(filename)

    def safe_load(self, filename):
        """ Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        """
        load_dict = dict()
        try:
            load_dict = self.load(filename)
            print(f'Model loaded: {filename}')
        except FileExistsError:
            print('Model NOT loaded')

        return load_dict

    def load_file(self, filename):
        """Loads a module dictionary from file.

        Args:
            filename (str): name of saved module dictionary
        """

        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print(filename)
            print('=> Loading checkpoint from local file...')
            state_dict = torch.load(filename)
            scalars = self.parse_state_dict(state_dict)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url):
        """Load a module dictionary from url.

        Args:
            url (str): url to saved model
        """
        print(url)
        print('=> Loading checkpoint from url...')
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict):
        """Parse state_dict of model and return scalars.

        Args:
            state_dict (dict): State dict of model
            """

        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print('Warning: Could not find %s in checkpoint!' % k)
        scalars = {k: v for k, v in state_dict.items()
                   if k not in self.module_dict}
        return scalars

