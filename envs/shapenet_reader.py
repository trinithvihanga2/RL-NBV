import os
import numpy as np
import open3d as o3d


class ShapenetReader:
    def __init__(self, data_path, logger):
        self.is_intial = False
        self.data_path = data_path
        self.logger = logger
        self.logger.info("ShapenetReader is ok")

        if not os.path.exists(self.data_path):
            self.logger.error("data path is not exist")
            return
        if not os.path.isdir(self.data_path):
            self.logger.error("data path is not a directory")
            return

        self.model_list = os.listdir(data_path)
        new_model_list = []
        for model in self.model_list:
            model_path = os.path.join(data_path, model)
            if os.path.isdir(model_path):
                new_model_list.append(model)
        self.model_list = new_model_list
        self.model_num = len(self.model_list)
        self.set_model_id(0)

        self.is_intial = True
        self.logger.info(
            "data path exists: {}, model num: {}".format(data_path, self.model_num)
        )

    def set_model_id(self, model_id):
        if model_id >= self.model_num or model_id < 0:
            self.logger.error("input model id: {} mistake".format(model_id))
            return False
        self.cur_model_id = model_id
        self.cur_model_name = self.model_list[model_id]
        ground_truth_path = os.path.join(
            self.data_path, self.model_list[model_id], "model.pcd"
        )
        self.ground_truth = o3d.io.read_point_cloud(ground_truth_path)
        self.ground_truth = np.array(self.ground_truth.points)
        return True

    def get_next_model(self):
        model_id = (self.cur_model_id + 1) % self.model_num
        self.set_model_id(model_id)

    def get_model_info(self):
        return self.cur_model_name
