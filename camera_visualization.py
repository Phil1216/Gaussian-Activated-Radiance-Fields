import matplotlib.pyplot as plt
import numpy as np
class camera_visualizer:
    def __init__(self, title = "Camera Extrinsic Parameters"):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        self.ax.clear()
        self.ax.set_title(title)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.lowerbond = 10000
        self.upperbond = -10000

    def get_wire_frame(self, scale = 1, GL=False):
        a = 0.5 * np.array([-2, 1.5, 4])
        up1 = 0.5 * np.array([0, 1.5, 4])
        up2 = 0.5 * np.array([0, 3, 4])
        b = 0.5 * np.array([2, 1.5, 4])
        c = 0.5 * np.array([-2, -1.5, 4])
        d = 0.5 * np.array([2, -1.5, 4])
        C = np.zeros(3)
        F = np.array([0, 0, 3])
        camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
        lines = np.stack([x.astype(np.float32) for x in camera_points])*scale
        if GL:
            lines[:,1] *= -1
            lines[:,2] *= -1
        return lines

    def wire_transpose(self,matrix, scale, GL=False):
        cam_wires_canonical = self.get_wire_frame(scale, GL)
        R = matrix[0:3,0:3]
        T = matrix[0:3,3]
        transposed_cam_wires = np.zeros_like(cam_wires_canonical)
        for i,wire in enumerate(cam_wires_canonical):
            transposed_cam_wires[i] = R @ wire + T
        x_, y_, z_ = transposed_cam_wires.T
        return x_, y_, z_

    def set_bound(self,xbond,ybond,zbond):
        self.ax.set_xlim3d(xbond)
        self.ax.set_ylim3d(ybond)
        self.ax.set_zlim3d(zbond)

    def plot_camera_scene(self, poses, scale=1, color: str = "blue",graph_label: str = "pose",GL = False):
        all = np.array([])
        for i in poses:
            x, y, z = self.wire_transpose(i, scale,GL)
            self.ax.plot(x,y,z, color=color, linewidth=0.5, label=graph_label)
            all = np.append(all,[x,y,z])
        
        lb = np.min(all)- np.absolute(np.min(all))*0.2
        ub = np.max(all) + np.absolute(np.max(all))*0.2
        if lb < self.lowerbond:
            self.lowerbond = lb
        if ub > self.upperbond:
            self.upperbond = ub
        self.set_bound([self.lowerbond,self.upperbond],[self.lowerbond,self.upperbond],[self.lowerbond,self.upperbond])
        self.ax.legend(bbox_to_anchor=(-0.40, 1), loc='upper left')
        #self.ax.legend(loc='outside left lower', bbox_to_anchor=(0, 0.5))
        self.fig.set_figwidth(20)
        self.fig.set_figheight(20)

    def show(self):
        plt.show()

    def save(self,path):
        plt.savefig(path)

    def set_title(self, newtitle):
        self.ax.set_title(newtitle)