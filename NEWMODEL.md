# How to create a VLM using new VISION Backbone

## Steps:

1. 在./prismatic/models/backbones/vision中创建并写好vision backbone（基于BaseVision）
2. 在./prismatic/models/backbones/vision/__init__.py中加入new vision backbone
3. 在./prismatic/models/materialize.py和registry.py中加入相关设置
4. 在./prismatic/conf/models.py中加入新模型的设置
