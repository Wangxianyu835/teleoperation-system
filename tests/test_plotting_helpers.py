import unittest
import numpy as np


class Axes:
    def set_xlim(self,*limits):self.x=limits
    def set_ylim(self,*limits):self.y=limits
    def set_zlim(self,*limits):self.z=limits


class PlottingHelpersTests(unittest.TestCase):
    def test_original_padding_and_degenerate_cloud_ranges_are_preserved(self):
        from teleoperation.tools.plotting import set_equal_axes
        for padding,radius in ((.55,1.),(.6,.1)):
            with self.subTest(padding=padding):
                axes=Axes();points=np.array([[0.,1.,2.],[2.,3.,6.]])
                set_equal_axes(axes,points,padding=padding,zero_radius=radius)
                half=4*padding
                for actual,center in zip((axes.x,axes.y,axes.z),(1.,2.,4.)):
                    np.testing.assert_allclose(actual,[center-half,center+half])
                set_equal_axes(axes,np.zeros((25,3)),padding=padding,zero_radius=radius)
                for limits in (axes.x,axes.y,axes.z):np.testing.assert_allclose(limits,[-radius,radius])
