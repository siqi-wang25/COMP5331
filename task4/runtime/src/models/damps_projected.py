"""SMORE adapter over the existing Task 1 shared DAMPS implementation.

The snapshot in common.damps_task1 is unchanged; see task4/README.md.
"""
from common.damps_task1 import DAMPS


class DAMPSProjected(DAMPS):
    def __init__(self, image_init, text_init, variant='full'):
        if variant not in ('full', 'no_apc', 'no_avrf', 'no_imcf'):
            raise ValueError(variant)
        super().__init__(image_init.shape[1], apc=variant != 'no_apc',
                         avrf=variant != 'no_avrf', imcf=variant != 'no_imcf')
        self.to(image_init)
        self.initialize(image_init, text_init)
