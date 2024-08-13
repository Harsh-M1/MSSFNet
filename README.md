# MSSFNet
MSSFNet and CHN-YE7-FRA dataset for Sentinel-2 Images Extraction of Offshore Floating Raft AquacultureThis repository includes the pytorch version of MSSFNet code and CHN-YE7-FRA dataset.
## MSSFNet
In MSSFNet, we innovatively design a spatial-spectral feature extraction block (SSFEB), which is used to replace the ResBlock in ResNet18 as the encoder of the model. The SSFEB is able to simultaneously extract multilevel spatial features and the links between spectral channels, realizing the efficient fusion of spatial and spectral information. It captures the complementary information between spectral bands and integrates it into the spatial domain, which enhances the comprehensiveness and richness of the feature representations. This design not only enhances the ability of the model to perceive FRA features in different bands but also improves the complementarity of the spectral and spatial features so that the model can accurately extract the target region even under complex backgrounds. In addition, we design a multiscale spatial attention block (MSAB) for capturing spatial features at different scales. The MSAB is able to integrate spatial information acquired under different receptive fields to realize a global receptive field and multiscale learning, which improves the ability of the model to detect various FRA features. By integrating multiscale features, the MSAB enhances the ability of the network to detect FRA regions of different sizes and shapes and improves its adaptability to complex backgrounds. Through the synergy of the SSFEB and MSAB, MSSFNet effectively integrates and utilizes multiscale spatial and spectral features, which significantly improves the extraction accuracy achieved for FRA regions in multispectral RSIs. This design not only optimizes the feature extraction process but also enhances the robustness and flexibility of the model in addressing complex marine environments so that it can better cope with variable ocean conditions and background interference and ultimately achieve high-precision FRA region extraction.

# CHN-YE7-FRA Dataset
The dataset consists of manually labelled pixel-level Sentinel-2 satellite images covering seven typical FRA regions in the Yellow Sea and East China Sea of China, and the data include various FRA types covering a variety of complex environmental and geographical conditions.
<div align=center>
<img src=https://github.com/user-attachments/assets/f6b90136-49b2-4f81-8b90-6b5ee49e88b3 width=80%/>
</div>

This dataset is a .tif format image and label file, due to the large file size, it can not be directly uploaded to github, so please download it through Google Cloud Drive or Baidu Cloud Drive, please pay attention to the citation source when using.

*Google Cloud Drive：*    
RGB tri-band synthesized version: https://drive.google.com/file/d/1WmWMYqPqV8oerl0OfjHDW-Abuk_AK4E0/view?usp=drive_link  
MSI version:  
*Baidu Cloud Drive：*    
RGB tri-band synthesized version: https://pan.baidu.com/s/1BaGtcSV2kAIPunYuyEKpCA?pwd=23tm  
MSI version:  
If you have any problems with use or download, you can contact : yhmhhxx@163.com ;

## Description of data
|Area	|Image Size	|Image Date |CoverageArea(km2) |
| ------ | ------ |------ |------ |
|Changhai County|	4450×2476|	2022.12.01 and 2023.01.25| 110.18 |
|Jinshitan Bay|	5879×2387|	2022.12.01 and 2023.01.25| 140.33|
|Rongcheng Bay|	3181×5927|	2022.12.01 and 2023.01.25| 188.54|
|Haizhou Bay|	4096×3568|	2022.12.05 and 2023.01.20| 146.15|
|Dayu Bay|	1065×902|	2022.12.20 and 2023.01.25| 9.61|
|Sansha Bay|	6101×4092|	2022.12.21 and 2023.01.25| 249.65|
|Zhaoan Bay|	1239×2121|	2022.12.05 and 2023.01.25| 26.28 |

** ! We are in the process of organizing and uploading the dataset, and will share it in the future via a link between Baidu Cloud Drive and Google Cloud Drive!
# Cite
## Chicago/Turabian Style
Yu, Haomiao, Yingzi Hou, Fangxiong Wang, Junfu Wang, Jianfeng Zhu, and Jianke Guo. 2024. "MSSFNet: A Multiscale Spatial–Spectral Fusion Network for Extracting Offshore Floating Raft Aquaculture Areas in Multispectral Remote Sensing Images" Sensors 24, no. 16: 5220. https://doi.org/10.3390/s24165220
## BibTex
@Article{  s24165220,  
AUTHOR = {Yu, Haomiao and Hou, Yingzi and Wang, Fangxiong and Wang, Junfu and Zhu, Jianfeng and Guo, Jianke},  
TITLE = {MSSFNet: A Multiscale Spatial–Spectral Fusion Network for Extracting Offshore Floating Raft Aquaculture Areas in Multispectral Remote Sensing Images},  
JOURNAL = {Sensors},  
VOLUME = {24},  
YEAR = {2024},  
NUMBER = {16},  
ARTICLE-NUMBER = {5220},  
URL = {https://www.mdpi.com/1424-8220/24/16/5220},  
ISSN = {1424-8220},  
DOI = {10.3390/s24165220}  
  }

