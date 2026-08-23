[Setup]
AppName = SQL_Manager
AppVersion = 2.1.1
AppPublisher = fhz
OutputDir = C:\Users\ASUS\Desktop\myself_app\SQL_Manager\output\Inno
OutputBaseFilename = SQL_Manager
SetupIconFile = C:\Users\ASUS\Desktop\myself_app\SQL_Manager\icon.ico
DefaultDirName=C:\Program Files\SQL_Manager
WizardStyle = modern
Compression = lzma
SolidCompression = yes
[Files]
; 打包根目录的app.exe，安装后放到安装目录根目录
Source: "C:\Users\ASUS\Desktop\myself_app\SQL_Manager\output\app\app.exe"; DestDir: "{app}"; Flags: ignoreversion

; 打包_internal整个文件夹的所有内容（包括子文件夹），安装后保留_internal文件夹结构
Source: "C:\Users\ASUS\Desktop\myself_app\SQL_Manager\output\app\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs