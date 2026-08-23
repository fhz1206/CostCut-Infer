[Setup]
AppName = CostCut Infer
AppVersion = 0.1.0_beta
AppPublisher = CostCut Infer
OutputDir = ..\setup
OutputBaseFilename = CostCutInfer-0.1.0_beta-setup
DefaultDirName=C:\Program Files\CostCutInfer
WizardStyle = modern
Compression = lzma
SolidCompression = yes
[Files]
; 打包 v0.1.0_beta 的 costcut-infer.exe，安装后放到安装目录根目录
Source: "v0.1.0_beta\costcut-infer.exe"; DestDir: "{app}"; Flags: ignoreversion

; 打包 v0.1.0_beta 的 dll（运行时依赖：torch_cpu/c10/fbgemm 等），安装后保留在安装目录
Source: "v0.1.0_beta\*.dll"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
