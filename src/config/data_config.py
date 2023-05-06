from typing import Dict, Any

from config.base_config import BaseConfig


class DataConfig(BaseConfig):
    required_arguments = {"train_file_path", "dev_file_path", "test_file_path",
                          "use_title", "mesh_filtering", "mesh_path", "saved_data_path"}

    def __init__(self,
                 train_file_path: str,
                 dev_file_path: str,
                 test_file_path: str,
                 use_title: bool,
                 mesh_filtering: bool,
                 mesh_path: str,
                 saved_data_path: str):
        self.train_file_path = train_file_path
        self.dev_file_path = dev_file_path
        self.test_file_path = test_file_path
        self.use_title = use_title
        self.mesh_filtering = mesh_filtering
        self.mesh_path = mesh_path
        self.saved_data_path = saved_data_path

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        cls.check_required(d)
        config = DataConfig(
            train_file_path=d['train_file_path'],
            dev_file_path=d['dev_file_path'],
            test_file_path=d['test_file_path'],
            use_title=d['use_title'],
            mesh_path=d['mesh_path'],
            mesh_filtering=d['mesh_filtering'],
            saved_data_path=d['saved_data_path']
        )
        return config



