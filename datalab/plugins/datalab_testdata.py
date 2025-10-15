# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test Data Plugin for DataLab
----------------------------

This plugin is an example of DataLab plugin. It provides test data samples
and some actions to test DataLab functionalities.
"""

from __future__ import annotations

import sigima.tests.data as test_data
from sigima.io.image import ImageIORegistry
from sigima.io.signal import SignalIORegistry
from sigima.tests import helpers

from datalab.config import _
from datalab.plugins import PluginBase, PluginInfo
from datalab.utils.qthelpers import create_progress_bar

# ------------------------------------------------------------------------------
# All computation functions must be defined as global functions, otherwise
# they cannot be pickled and sent to the worker process
# ------------------------------------------------------------------------------


class PluginTestData(PluginBase):
    """DataLab Test Data Plugin"""

    PLUGIN_INFO = PluginInfo(
        name=_("Test data"),
        version="1.0.0",
        description=_("Testing DataLab functionalities"),
    )

    def load_test_objs(
        self, registry_class: type[SignalIORegistry | ImageIORegistry], title: str
    ) -> None:
        """Load all test objects from a given registry class

        Args:
            registry_class: Registry class (SignalIORegistry or ImageIORegistry)
            title: Progress bar title

        Returns:
            List of (filename, object) tuples
        """
        test_objs = list(helpers.read_test_objects(registry_class))
        with create_progress_bar(self.signalpanel, title, max_=len(test_objs)) as prog:
            for i_obj, (_fname, obj) in enumerate(test_objs):
                prog.setValue(i_obj + 1)
                if prog.wasCanceled():
                    break
                if obj is not None:
                    self.proxy.add_object(obj)

    # Signal processing features ------------------------------------------------
    def create_paracetamol_signal(self) -> None:
        """Create paracetamol signal"""
        obj = test_data.create_paracetamol_signal()
        self.proxy.add_object(obj)

    # Image processing features ------------------------------------------------
    def create_peak_image(self) -> None:
        """Create 2D peak image"""
        obj = self.imagepanel.new_object(add_to_panel=False)
        if obj is not None:
            param = test_data.PeakDataParam.create(size=max(obj.data.shape))
            self.imagepanel.processor.update_param_defaults(param)
            if param.edit(self.main):
                obj.data, _coords = test_data.get_peak2d_data(param)
                self.proxy.add_object(obj)

    def create_sincos_image(self) -> None:
        """Create 2D sin cos image"""
        newparam = self.edit_new_image_parameters(hide_type=True)
        if newparam is not None:
            obj = test_data.create_sincos_image(newparam)
            self.proxy.add_object(obj)

    def create_noisy_gaussian_image(self) -> None:
        """Create 2D noisy gauss image"""
        newparam = self.edit_new_image_parameters(hide_height=True, hide_type=True)
        if newparam is not None:
            obj = test_data.create_noisy_gaussian_image(newparam, add_annotations=False)
            self.proxy.add_object(obj)

    def create_multigaussian_image(self) -> None:
        """Create 2D multi gauss image"""
        newparam = self.edit_new_image_parameters(hide_height=True, hide_type=True)
        if newparam is not None:
            obj = test_data.create_multigaussian_image(newparam)
            self.proxy.add_object(obj)

    def create_2dstep_image(self) -> None:
        """Create 2D step image"""
        newparam = self.edit_new_image_parameters(hide_type=True)
        if newparam is not None:
            obj = test_data.create_2dstep_image(newparam)
            self.proxy.add_object(obj)

    def create_ring_image(self) -> None:
        """Create 2D ring image"""
        param = test_data.RingParam(_("Ring"))
        if param.edit(self.main):
            obj = test_data.create_ring_image(param)
            self.proxy.add_object(obj)

    def create_annotated_image(self) -> None:
        """Create annotated image"""
        obj = test_data.create_annotated_image()
        self.proxy.add_object(obj)

    def create_grid_gaussian_image(self) -> None:
        """Create image with a grid of gaussian spots"""
        param = test_data.GridOfGaussianImages(_("Grid of Gaussian Images"))
        if param.edit(self.main):
            obj = test_data.create_grid_of_gaussian_images(param)
            self.proxy.add_object(obj)

    # Plugin menu entries ------------------------------------------------------
    def create_actions(self) -> None:
        """Create actions"""
        # Signal Panel ----------------------------------------------------------
        sah = self.signalpanel.acthandler
        with sah.new_menu(_("Test data")):
            sah.new_action(
                _("Load spectrum of paracetamol"),
                triggered=self.create_paracetamol_signal,
                select_condition="always",
            )
            sah.new_action(
                _("Load all test signals"),
                triggered=lambda regclass=SignalIORegistry,
                title=_("Load all test signals"): self.load_test_objs(regclass, title),
                select_condition="always",
                separator=True,
            )
        # Image Panel -----------------------------------------------------------
        iah = self.imagepanel.acthandler
        with iah.new_menu(_("Test data")):
            iah.new_action(
                _("Create image with peaks"),
                triggered=self.create_peak_image,
                select_condition="always",
                separator=True,
            )
            iah.new_action(
                _("Create 2D sin cos image"),
                triggered=self.create_sincos_image,
                select_condition="always",
            )
            iah.new_action(
                _("Create 2D noisy gaussian image"),
                triggered=self.create_noisy_gaussian_image,
                select_condition="always",
            )
            iah.new_action(
                _("Create 2D multi gaussian image"),
                triggered=self.create_multigaussian_image,
                select_condition="always",
            )
            iah.new_action(
                _("Create annotated image"),
                triggered=self.create_annotated_image,
                select_condition="always",
            )
            iah.new_action(
                _("Create 2D step image"),
                triggered=self.create_2dstep_image,
                select_condition="always",
            )
            iah.new_action(
                _("Create ring image"),
                triggered=self.create_ring_image,
                select_condition="always",
            )
            iah.new_action(
                _("Create image with a grid of gaussian spots"),
                triggered=self.create_grid_gaussian_image,
                select_condition="always",
            )
            iah.new_action(
                _("Load all test images"),
                triggered=lambda regclass=ImageIORegistry,
                title=_("Load all test images"): self.load_test_objs(regclass, title),
                select_condition="always",
                separator=True,
            )
