# Comparisons among different anime models

[English](anime_comparisons.md) **|** [简体中文](anime_comparisons_CN.md)

## Update News

- 2022/04/24: Release **AnimeVideo-v3**. We have made the following improvements:
  - **better naturalness**
  - **Fewer artifacts**
  - **more faithful to the original colors**
  - **better texture restoration**
  - **better background restoration**

## Comparisons

We have compared our RealESRGAN-AnimeVideo-v3 with the following methods.
Our RealESRGAN-AnimeVideo-v3 can achieve better results with faster inference speed.

- [waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan) with the hyperparameters: `tile=0`, `noiselevel=2`
- [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN): we use the [20220227](https://github.com/bilibili/ailab/releases/tag/Real-CUGAN-add-faster-low-memory-mode) version, the hyperparameters are: `cache_mode=0`, `tile=0`, `alpha=1`.
- our RealESRGAN-AnimeVideo-v3

## Results

You may need to **zoom in** for comparing details, or **click the image** to see in the full size. Please note that the images
in the table below are the resized and cropped patches from the original images, you can download the original inputs and outputs from [Google Drive](https://drive.google.com/drive/folders/1bc_Hje1Nqop9NDkUvci2VACSjL7HZMRp?usp=sharing) .

**More natural results, better background restoration**
| Input | waifu2x | Real-CUGAN | RealESRGAN<br>AnimeVideo-v3 |
| :---: | :---:        |     :---:      |  :---:      |
|![157083983-bec52c67-9a5e-4eed-afef-01fe6cd2af85_patch](https://user-images.githubusercontent.com/11482921/164452769-5d8cb4f8-1708-42d2-b941-f44a6f136feb.png) | ![](https://user-images.githubusercontent.com/11482921/164452767-c825cdec-f721-4ff1-aef1-fec41f146c4c.png) | ![](https://user-images.githubusercontent.com/11482921/164452755-3be50895-e3d4-432d-a7b9-9085c2a8e771.png) | ![](https://user-images.githubusercontent.com/11482921/164452771-be300656-379a-4323-a755-df8025a8c451.png) |
|![a0010_patch](https://user-images.githubusercontent.com/11482921/164454047-22eeb493-3fa9-4142-9fc2-6f2a1c074cd5.png) | ![](https://user-images.githubusercontent.com/11482921/164454046-d5e79f8f-00a0-4b55-bc39-295d0d69747a.png) | ![](https://user-images.githubusercontent.com/11482921/164454040-87886b11-9d08-48bd-862f-0d4aed72eb19.png) | ![](https://user-images.githubusercontent.com/11482921/164454055-73dc9f02-286e-4d5c-8f70-c13742e08f42.png) |
|![00000044_patch](https://user-images.githubusercontent.com/11482921/164451232-bacf64fc-e55a-44db-afbb-6b31ab0f8973.png) | ![](https://user-images.githubusercontent.com/11482921/164451318-f309b61a-75b8-4b74-b5f3-595725f1cf0b.png) | ![](https://user-images.githubusercontent.com/11482921/164451348-994f8a35-adbe-4a4b-9c61-feaa294af06a.png) | ![](https://user-images.githubusercontent.com/11482921/164451361-9b7d376e-6f75-4648-b752-542b44845d1c.png) |

**Fewer artifacts, better detailed textures**
| Input | waifu2x | Real-CUGAN | RealESRGAN<br>AnimeVideo-v3 |
| :---: | :---:        |     :---:      |  :---:      |
|![00000053_patch](https://user-images.githubusercontent.com/11482921/164448411-148a7e5c-cfcd-4504-8bc7-e318eb883bb6.png) | ![](https://user-images.githubusercontent.com/11482921/164448633-dfc15224-b6d2-4403-a3c9-4bb819979364.png) | ![](https://user-images.githubusercontent.com/11482921/164448771-0d359509-5293-4d4c-8e3c-86a2a314ea88.png) | ![](https://user-images.githubusercontent.com/11482921/164448848-1a4ff99e-075b-4458-9db7-2c89e8160aa0.png) |
|![Disney_v4_22_018514_s2_patch](https://user-images.githubusercontent.com/11482921/164451898-83311cdf-bd3e-450f-b9f6-34d7fea3ab79.png) | ![](https://user-images.githubusercontent.com/11482921/164451894-6c56521c-6561-40d6-a3a5-8dde2c167b8a.png) | ![](https://user-images.githubusercontent.com/11482921/164451888-af9b47e3-39dc-4f3e-b0d7-d372d8191e2a.png) | ![](https://user-images.githubusercontent.com/11482921/164451901-31ca4dd4-9847-4baa-8cde-ad50f4053dcf.png) |
|![Japan_v2_0_007261_s2_patch](https://user-images.githubusercontent.com/11482921/164454578-73c77392-77de-49c5-b03c-c36631723192.png) | ![](https://user-images.githubusercontent.com/11482921/164454574-b1ede5f0-4520-4eaa-8f59-086751a34e62.png) | ![](https://user-images.githubusercontent.com/11482921/164454567-4cb3fdd8-6a2d-4016-85b2-a305a8ff80e4.png) | ![](https://user-images.githubusercontent.com/11482921/164454583-7f243f20-eca3-4500-ac43-eb058a4a101a.png) |
|![huluxiongdi_2_patch](https://user-images.githubusercontent.com/11482921/164453482-0726c842-337e-40ec-bf6c-f902ee956a8b.png) | ![](https://user-images.githubusercontent.com/11482921/164453480-71d5e091-5bfa-4c77-9c57-4e37f66ca0a3.png) | ![](https://user-images.githubusercontent.com/11482921/164453468-c295d3c9-3661-45f0-9ecd-406a1877f76e.png) | ![](https://user-images.githubusercontent.com/11482921/164453486-3091887c-587c-450e-b6fe-905cb518d57e.png) |

**Other better results**
| Input | waifu2x | Real-CUGAN | RealESRGAN<br>AnimeVideo-v3 |
| :---: | :---:        |     :---:      |  :---:      |
|![Japan_v2_1_128525_s1_patch](https://user-images.githubusercontent.com/11482921/164454933-67697f7c-b6ef-47dc-bfca-822a78af8acf.png) | ![](https://user-images.githubusercontent.com/11482921/164454931-9450de7c-f0b3-4638-9c1e-0668e0c41ef0.png) | ![](https://user-images.githubusercontent.com/11482921/164454926-ed746976-786d-41c5-8a83-7693cd774c3a.png) | ![](https://user-images.githubusercontent.com/11482921/164454936-8abdf0f0-fb30-40eb-8281-3b46c0bcb9ae.png) |
|![tianshuqitan_2_patch](https://user-images.githubusercontent.com/11482921/164456948-807c1476-90b6-4507-81da-cb986d01600c.png) | ![](https://user-images.githubusercontent.com/11482921/164456943-25e89de9-d7e5-4f61-a2e1-96786af6ae9e.png) | ![](https://user-images.githubusercontent.com/11482921/164456954-b468c447-59f5-4594-9693-3683e44ba3e6.png) | ![](https://user-images.githubusercontent.com/11482921/164456957-640f910c-3b04-407c-ac20-044d72e19735.png) |
|![00000051_patch](https://user-images.githubusercontent.com/11482921/164456044-e9a6b3fa-b24e-4eb7-acf9-1f7746551b1e.png) ![00000051_patch](https://user-images.githubusercontent.com/11482921/164456421-b67245b0-767d-4250-9105-80bbe507ecfc.png) | ![](https://user-images.githubusercontent.com/11482921/164456040-85763cf2-cb28-4ba3-abb6-1dbb48c55713.png) ![](https://user-images.githubusercontent.com/11482921/164456419-59cf342e-bc1e-4044-868c-e1090abad313.png) | ![](https://user-images.githubusercontent.com/11482921/164456031-4244bb7b-8649-4e01-86f4-40c2099c5afd.png) ![](https://user-images.githubusercontent.com/11482921/164456411-b6afcbe9-c054-448d-a6df-96d3ba3047f8.png) | ![](https://user-images.githubusercontent.com/11482921/164456035-12e270be-fd52-46d4-b18a-3d3b680731fe.png) ![](https://user-images.githubusercontent.com/11482921/164456417-dcaa8b62-f497-427d-b2d2-f390f1200fb9.png) |
|![00000099_patch](https://user-images.githubusercontent.com/11482921/164455312-6411b6e1-5823-4131-a4b0-a6be8a9ae89f.png) | ![](https://user-images.githubusercontent.com/11482921/164455310-f2b99646-3a22-47a4-805b-dc451ac86ddb.png) | ![](https://user-images.githubusercontent.com/11482921/164455294-35471b42-2826-4451-b7ec-6de01344954c.png) | ![](https://user-images.githubusercontent.com/11482921/164455305-fa4c9758-564a-4081-8b4e-f11057a0404d.png) |
|![00000016_patch](https://user-images.githubusercontent.com/11482921/164455672-447353c9-2da2-4fcb-ba4a-7dd6b94c19c1.png) | ![](https://user-images.githubusercontent.com/11482921/164455669-df384631-baaa-42f8-9150-40f658471558.png) | ![](https://user-images.githubusercontent.com/11482921/164455657-68006bf0-138d-4981-aaca-8aa927d2f78a.png) | ![](https://user-images.githubusercontent.com/11482921/164455664-0342b93e-a62a-4b36-a90e-7118f3f1e45d.png) |

## Inference Speed

### PyTorch

Note that we only report the **model** time, and ignore the IO time.

| GPU | Input Resolution | waifu2x | Real-CUGAN | RealESRGAN-AnimeVideo-v3
| :---: | :---:         |  :---:        |     :---:      |  :---:      |
| V100 | 1921 x 1080 | - | 3.4 fps | **10.0** fps |
| V100 | 1280 x 720 | - | 7.2 fps | **22.6** fps |
| V100 | 640 x 480 | - | 24.4 fps | **65.9** fps |

### ncnn

- [ ] TODO
