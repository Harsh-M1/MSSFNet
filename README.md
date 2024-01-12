# MVMSA-Net
MVMSA-Net and CHN-YE7-FRA dataset for Sentinel-2 Images Extraction of Offshore Floating Raft AquacultureThis repository includes the pytorch version of MVMSA-Net code and CHN-YE7-FRA dataset.
## MVMSA-Net
The overall structure of our MVMSA-Net is designed based on UNet, and a combination of MobileViT Block and MV2 Block is used to enable the model to effectively extract and process global and local semantic information. And the Multi-scale Spatial Awareness Block (MSA) is designed to mitigate the effect of FRA size differences and variability in different regions on the extraction accuracy. In order to minimize the loss of FRA semantic information due to bilinear interpolation upsampling, we introduce Dysample upsampling in the network.
## CHN-YE7-FRA
We constructed a dataset CHN-YE7-FRA Dataset based on Sentinel-2 remote sensing images containing seven typical coastal FRA regions in the Yellow Sea and the East China Sea of China.The dataset images were synthesized using the three bands of R\G\B, and the FRA regions in the images were labeled at pixel level by professional interpretation experts.
