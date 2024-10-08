from typing import Any, Dict, List, Optional, Type, Union
import os
import time
import datetime

import vtk
# import vtkmodules.all as vtk


vtk_color = {
    'Whites': ['antique_white', 'azure', 'bisque', 'blanched_almond',
                'cornsilk', 'eggshell', 'floral_white', 'gainsboro',
                'ghost_white', 'honeydew', 'ivory', 'lavender',
                'lavender_blush', 'lemon_chiffon', 'linen', 'mint_cream',
                'misty_rose', 'moccasin', 'navajo_white', 'old_lace',
                'papaya_whip', 'peach_puff', 'seashell', 'snow',
                'thistle', 'titanium_white', 'wheat', 'white',
                'white_smoke', 'zinc_white'],
    'Greys': ['cold_grey', 'dim_grey', 'grey', 'light_grey',
                'slate_grey', 'slate_grey_dark', 'slate_grey_light',
                'warm_grey'],
    'Reds': ['coral', 'coral_light', 
                'hot_pink', 'light_salmon',
                'pink', 'pink_light',
                'raspberry', 'rose_madder', 'salmon',
                ],
    # 'Browns': ['beige', 'brown', 'brown_madder', 'brown_ochre',
    #             'burlywood', 'burnt_sienna', 'burnt_umber', 'chocolate',
    #             'flesh', 'flesh_ochre', 'gold_ochre',
    #             'greenish_umber', 'khaki', 'khaki_dark', 'light_beige',
    #             'peru', 'rosy_brown', 'raw_sienna', 'raw_umber', 'sepia',
    #             'sienna', 'saddle_brown', 'sandy_brown', 'tan',
    #             'van_dyke_brown'],
    'Oranges': ['cadmium_orange', 'cadmium_red_light', 'carrot',
                'dark_orange', 'mars_orange', 'mars_yellow', 'orange',
                'orange_red', 'yellow_ochre'],
    'Yellows': ['aureoline_yellow', 'banana', 'cadmium_lemon',
                'cadmium_yellow', 'cadmium_yellow_light', 'gold',
                'goldenrod', 'goldenrod_dark', 'goldenrod_light',
                'goldenrod_pale', 'light_goldenrod', 'melon',
                'yellow', 'yellow_light'],
    'Greens': ['chartreuse', 'chrome_oxide_green', 'cinnabar_green',
                'cobalt_green', 'emerald_green', 'forest_green', 
                'green_dark', 'green_pale', 'green_yellow', 'lawn_green',
                'lime_green', 'mint', 'olive', 'olive_drab',
                'olive_green_dark', 'permanent_green', 'sap_green',
                'sea_green', 'sea_green_dark', 'sea_green_medium',
                'sea_green_light', 'spring_green', 'spring_green_medium',
                'terre_verte', 'viridian_light', 'yellow_green'],
    'Cyans': ['aquamarine', 'aquamarine_medium', 'cyan', 'cyan_white',
                'turquoise', 'turquoise_dark', 'turquoise_medium',
                'turquoise_pale'],
    'Blues': ['alice_blue', 'blue_light', 'blue_medium',
                'cadet', 'cobalt', 'cornflower', 'cerulean', 'dodger_blue',
                'indigo', 'manganese_blue', 'midnight_blue', 'navy',
                'peacock', 'powder_blue', 'royal_blue', 'slate_blue',
                'slate_blue_dark', 'slate_blue_light',
                'slate_blue_medium', 'sky_blue', 
                'sky_blue_light', 'steel_blue', 'steel_blue_light',
                'turquoise_blue', 'ultramarine'],
    'Magentas': ['blue_violet', 'magenta',
                    'orchid', 'orchid_dark', 'orchid_medium',
                    'plum', 'purple',
                    'purple_medium', 'ultramarine_violet', 'violet',
                    'violet_dark', 'violet_red_medium',
                    'violet_red_pale']
}
color_key = list(vtk_color.keys())


class VTKRender:
    def __init__(
            self, 
            container_size: List[int], 
            win_size: List[int]=[600, 600], 
            offscreen: bool=True,
            auto_render: bool=True
        ) -> None:
        self.container_size = container_size
        self.item_idx = 0
        self.auto_render = auto_render

        # 1. render
        self.render = vtk.vtkRenderer()    
        self.render.SetBackground(1.0, 1.0, 1.0)

        # 2. render window
        self.render_window = vtk.vtkRenderWindow()
        # if offscreen:
        #     self.render_window.SetOffScreenRendering(1)
        self.render_window.SetWindowName("Packing Visualization")
        self.render_window.SetSize(win_size[0], win_size[1])
        self.render_window.AddRenderer(self.render)

        # 3. interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        # 4. camera
        self.camera = vtk.vtkCamera()
        self.camera.SetPosition(
            2.5 * max(self.container_size), 
            -2 * max(self.container_size), 
            2 * max(self.container_size)
        )
        self.camera.SetViewUp(0, 0, 1.5 * max(container_size))
        self.render.SetActiveCamera(self.camera)

        # 5. axes
        self._init_axes()

        # 6. container (cube)
        self._init_container()

        self.interactor.Initialize()
        self.render_window.Render()
        time.sleep(0.5)

    def _init_axes(self) -> None:
        axes = vtk.vtkAxesActor()

        transform = vtk.vtkTransform()
        transform.Translate(
            -0.5 * self.container_size[0], 
            -0.5 * self.container_size[1], 
            -0.5 * self.container_size[2]
        )
        
        axes.SetUserTransform(transform)

        sigma = 0.1
        axes_l_x = self.container_size[0] + sigma * self.container_size[2]
        axes_l_y = self.container_size[1] + sigma * self.container_size[2]
        axes_l_z = (1 + sigma) * self.container_size[2]
        
        axes.SetTotalLength(axes_l_x, axes_l_y, axes_l_z)
        axes.SetNormalizedShaftLength(1, 1, 1)
        axes.SetNormalizedTipLength(0.05, 0.05, 0.05)
        axes.AxisLabelsOff()

        self.render.AddActor(axes)
    
    def _init_container(self) -> None:
        container = vtk.vtkCubeSource()
        container.SetXLength(self.container_size[0])
        container.SetYLength(self.container_size[1])
        container.SetZLength(self.container_size[2])
        container.SetCenter([0, 0, 0])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(container.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        actor.GetProperty().SetRepresentationToWireframe()
        
        self.render.AddActor(actor)

    def add_item(self, item_size: List[int], item_pos: List[int], dir: str="") -> None:

        item = vtk.vtkCubeSource()
        item.SetXLength(item_size[0])
        item.SetYLength(item_size[1])
        item.SetZLength(item_size[2])
        item.SetCenter([
            -0.5 * self.container_size[0] + 0.5 * item_size[0] + item_pos[0],
            -0.5 * self.container_size[1] + 0.5 * item_size[1] + item_pos[1],
            -0.5 * self.container_size[2] + 0.5 * item_size[2] + item_pos[2]
        ])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(item.GetOutputPort())
        
        colors = vtk.vtkNamedColors()
        color_0 = color_key[self.item_idx % len(color_key)]
        color_1 = int(self.item_idx / len(color_key))

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d("red"))
        actor.GetProperty().EdgeVisibilityOn()
        
        self.render.AddActor(actor)
        time.sleep(0.5)
        self.render_window.Render()
        
        time.sleep(0.3)
        actor.GetProperty().SetColor(colors.GetColor3d(vtk_color[color_0][color_1]))
        self.render_window.Render()
        
        self.item_idx += 1

        if not self.auto_render:
            self.hold_on()
    
    def hold_on(self) -> None:
        self.interactor.Start()

    def save_img(self) -> None:
        time_str = datetime.datetime.now().strftime("%Y.%m.%d-%H-%M-%S.%f")
        img_name = time_str + r".png"
        path = os.path.join("images", "tmp")
        if not os.path.exists(path):
            os.makedirs(path)

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(os.path.join(path, img_name))
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()


if __name__ == "__main__":
    render = VTKRender([10, 10, 10])

    render.add_item([2, 3, 2], [0, 0, 0])
    render.hold_on()
    render.add_item([1, 1, 1], [2, 0, 0])

    render.hold_on()
