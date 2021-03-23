clear;

load Phantom_truth.mat
load Phantom_noise10.mat
%% parameters


groundtruth = double(Phantom_truth(:,:,60));
Noised_phantom = double(Phantom_noise(:,:,60));

sigma_den = 0.9;  % 

%% graph parameters
param.nnparam.k = 10;
param.patch = 5;
param.nnparam.sigma = sigma_den*255;

%% contruct Graph
[G, nopixels, nopatches] = gsp_patch_graph(Noised_phantom,param);
        
%% graph spectral decomposed
G = gsp_compute_fourier_basis(G);
%% graph spectral filter
tau1 = 1;
h1 = @(x) 1./(1+(tau1)*(x.^3));
%% graph denoising
z1 = reshape(Noised_phantom,[G.N,1]);
f1 = gsp_filter(G,h1,z1);
Graph_denoise = reshape(f1,[size(Noised_phantom,1),size(Noised_phantom,2)]);
%% BM3D
[~,BM3D_denoise] = BM3D(1, Noised_phantom, sigma_den*255,'np',0);

%% NLM
NLM_denoise=NLmeans(Noised_phantom,1,3,sigma_den*255);




%% Result
Km = [0.01 0.03];
window = fspecial('gaussian', 13, 1.5);
L = max(groundtruth(:));
ssim_noisedPET = ssim1(groundtruth,Noised_phantom,Km,window,L);
ssim_NLM = ssim1(groundtruth,NLM_denoise,Km,window,L);
ssim_BM = ssim1(groundtruth,BM3D_denoise,Km,window,L);
ssim_GF = ssim1(groundtruth,Graph_denoise,Km,window,L);

gmin = 0;
gmax = 1.1*max(groundtruth(:));
grange = [gmin,gmax];

figure;
subplot(1,5,1),imagesc(groundtruth,grange),axis square, axis off;
title(['Truth image '])
subplot(1,5,2),imagesc(Noised_phantom,grange) ,colormap  , axis square, axis off;
title(['Noised image SSIM =',num2str(ssim_noisedPET)]);
subplot(1,5,3),imagesc(NLM_denoise,grange),colormap  ,axis square, axis off;
title(['NLM  SSIM =',num2str(ssim_NLM)]);    
subplot(1,5,4),imagesc(BM3D_denoise,grange),colormap  ,axis square, axis off;
title(['BM3D  SSIM = ',num2str(ssim_BM)]);
subplot(1,5,5),imagesc(Graph_denoise,grange),colormap  ,axis square, axis off;
title(['Graph filtering SSIM =',num2str(ssim_GF)]);
