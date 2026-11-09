; Inno Setup 6 script for XSTARS Windows WPS standalone distribution
; Produces XSTARS_WPS_Setup.exe for Windows 10/11 x64 (WPS 365/12.x)

#ifndef AppVersion
#define AppVersion "1.3.0"
#endif

[Setup]
AppId={{C5D076E3-8E5A-4B2D-9B8A-7D9A4C0B6A8E}}
AppName=XSTARS for WPS
AppVersion={#AppVersion}
AppVerName=XSTARS for WPS {#AppVersion}
AppPublisher=Frankkk1912
AppPublisherURL=https://github.com/Frankkk1912/xstars
AppSupportURL=https://github.com/Frankkk1912/xstars/issues
AppUpdatesURL=https://github.com/Frankkk1912/xstars/releases
DefaultDirName={localappdata}\XSTARS-WPS
DefaultGroupName=XSTARS for WPS
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=output
OutputBaseFilename=XSTARS_WPS_Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=XSTARS for WPS

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Default.isl"

[Files]
; Frozen service, worker, and helper onedir bundle
Source: "dist\xstars-wps\*"; DestDir: "{app}\service"; Flags: ignoreversion recursesubdirs createallsubdirs
; Official wps-addon offline publish and deployment artifacts
Source: "..\..\wps-addon\deploy\*"; DestDir: "{app}\addin\deploy"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\wps-addon\config.template.js"; DestDir: "{app}\addin"; Flags: ignoreversion
; User documentation
Source: "..\..\docs\wps-installation.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\..\README.zh-CN.md"; DestDir: "{app}\docs"; Flags: ignoreversion

[Icons]
Name: "{group}\安装或管理 WPS 加载项"; Filename: "{app}\service\xstars-wps-helper.exe"; Parameters: "install-page --dir ""{app}\addin\deploy"""; WorkingDir: "{app}\service"; IconFilename: "{app}\service\xstars-wps-helper.exe"
Name: "{group}\同步 WPS 加载项配置"; Filename: "{app}\service\xstars-wps-helper.exe"; Parameters: "sync-config --template ""{app}\addin\config.template.js"""; WorkingDir: "{app}\service"
Name: "{group}\XSTARS WPS 后台服务"; Filename: "{app}\service\xstars-wps.exe"; Parameters: "serve"; WorkingDir: "{app}\service"
Name: "{group}\安装与使用说明"; Filename: "{app}\docs\wps-installation.md"
Name: "{group}\卸载 XSTARS for WPS"; Filename: "{uninstallexe}"

[Registry]
; Service autostart on user logon (HKCU Run)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "XSTARS_WPS_Service"; ValueData: """{app}\service\xstars-wps.exe"" serve"; Flags: uninsdeletevalue

[Run]
Filename: "{app}\service\xstars-wps.exe"; Parameters: "serve"; Description: "启动 XSTARS WPS 本地后台服务"; Flags: nowait postinstall skipifsilent
Filename: "{app}\service\xstars-wps-helper.exe"; Parameters: "install-page --dir ""{app}\addin\deploy"""; Description: "打开 WPS 加载项安装页面 (推荐首次安装勾选)"; Flags: nowait postinstall unchecked skipifsilent

[UninstallRun]
Filename: "taskkill.exe"; Parameters: "/F /IM xstars-wps.exe /IM xstars-wps-helper.exe"; Flags: runhidden; RunOnceId: "StopXstarsWpsProcesses"

[Code]
// The background service has no window, so Restart Manager cannot close it
// and an update fails with "DeleteFile failed; code 5". Stop it (and the
// helper) before any file is replaced; [Run] or the logon entry restarts it.
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /T /IM xstars-wps.exe /IM xstars-wps-helper.exe',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  // Give Windows a moment to release the file handles.
  Sleep(1000);
  Result := '';
end;

// Execute helper actions during installation: backup, bootstrap, and initial sync
procedure CurStepChanged(CurStep: TSetupStep);
var
  HelperExe: String;
  ServiceExe: String;
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    HelperExe := ExpandConstant('{app}\service\xstars-wps-helper.exe');
    if FileExists(HelperExe) then
    begin
      // 1. Back up any existing WPS jsaddons config to %LOCALAPPDATA%\XSTARS-WPS\backup\<timestamp>
      Exec(HelperExe, 'backup', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

      // 2. Bootstrap per-install secret token and render config.js
      Exec(HelperExe, ExpandConstant('bootstrap --template "{app}\addin\config.template.js" --out "{app}\addin\config.js" "{app}\addin\deploy\config.js"'), '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

      // 3. Sync config to any currently installed WPS add-in directory
      Exec(HelperExe, ExpandConstant('sync-config --template "{app}\addin\config.template.js"'), '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    end;

    // [Run] skips the service in silent installs; restart it so a silent
    // update does not leave WPS without its service until the next logon.
    ServiceExe := ExpandConstant('{app}\service\xstars-wps.exe');
    if WizardSilent and FileExists(ServiceExe) then
      Exec(ServiceExe, 'serve', ExpandConstant('{app}\service'), SW_HIDE, ewNoWait, ResultCode);
  end;
end;
