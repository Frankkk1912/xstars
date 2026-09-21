; XSTARS Inno Setup Script
; Produces: XSTARS_Setup_v1.2.0.exe
;
; Build prerequisites:
;   1. PyInstaller output in dist\xstars\
;   2. XSTARS.xlam in installer\ (built by build_installer.py)
;   3. Inno Setup 6 installed (https://jrsoftware.org/isinfo.php)
;
; Build:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\XSTARS.iss

#define MyAppName "XSTARS"
#define MyAppVersion "1.2.0"
#define MyAppPublisher "Frank-SYSU"
#define MyAppURL "https://github.com/Frankkk1912/xstars"

; All Source paths are relative to this root
; NOTE: this .iss lives in installer\excel\, so the project root is two levels up
#define ProjectRoot "..\.."

[Setup]
AppId={{E7A3B2C1-4D5F-6789-ABCD-EF0123456789}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} v{#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=XSTARS_Setup_v{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
; Allow non-admin install (installs to %LOCALAPPDATA%\Programs\XSTARS)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
; SetupIconFile=
UninstallDisplayIcon={app}\xstars\xstars.exe
; Minimum Windows version: Windows 10
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Python backend (PyInstaller one-dir output)
Source: "{#ProjectRoot}\dist\xstars\*"; DestDir: "{app}\xstars"; \
    Flags: recursesubdirs ignoreversion

; Excel add-in -> user's standard AddIns folder
Source: "XSTARS.xlam"; DestDir: "{userappdata}\Microsoft\AddIns"; \
    Flags: ignoreversion

; Registration scripts -> install dir (for uninstall)
Source: "register_addin.vbs"; DestDir: "{app}"; Flags: ignoreversion
Source: "unregister_addin.vbs"; DestDir: "{app}"; Flags: ignoreversion

[Registry]
; Store install path so VBA ExePath() can locate the exe
Root: HKCU; Subkey: "Software\XSTARS"; \
    ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; \
    Flags: uninsdeletekey

[UninstallDelete]
; Clean up the xlam from AddIns folder
Type: files; Name: "{userappdata}\Microsoft\AddIns\XSTARS.xlam"

[UninstallRun]
; Pre-uninstall: deregister the add-in via Pascal script
Filename: "wscript.exe"; \
    Parameters: """{app}\unregister_addin.vbs"" ""{userappdata}\Microsoft\AddIns\XSTARS.xlam"""; \
    RunOnceId: "UnregAddin"; \
    Flags: runhidden waituntilterminated

[Messages]
FinishedLabel=XSTARS has been installed successfully.%n%nOpen any Excel file and look for the "XSTARS" tab in the ribbon.%n%nIf the tab does not appear, go to Excel > File > Options > Add-ins > Manage: Excel Add-ins > Go > check "XSTARS".

[Code]
// Register the .xlam as an Excel add-in by writing to the registry.
// Excel loads add-ins listed under:
//   HKCU\Software\Microsoft\Office\<ver>\Excel\Options
//   OPEN, OPEN1, OPEN2, ...  (value = /R "path\to\addin.xlam")
//
// We scan Office versions 14.0..16.0 (Excel 2010 through 365).

const
  OfficeBase = 'Software\Microsoft\Office\';

// Find next available OPENn slot and write our add-in path
procedure RegisterAddinForVersion(Version: String; XlamPath: String);
var
  SubKey, ValueName, Existing: String;
  I: Integer;
  Found: Boolean;
begin
  SubKey := OfficeBase + Version + '\Excel\Options';

  // Check if this Office version's Excel key exists at all
  if not RegKeyExists(HKCU, SubKey) then
    Exit;

  // Check if already registered (avoid duplicates)
  // First check OPEN (no number)
  if RegQueryStringValue(HKCU, SubKey, 'OPEN', Existing) then
  begin
    if Pos(UpperCase('XSTARS.xlam'), UpperCase(Existing)) > 0 then
      Exit;  // already registered
  end
  else
  begin
    // OPEN doesn't exist — use it
    RegWriteStringValue(HKCU, SubKey, 'OPEN', '/R "' + XlamPath + '"');
    Exit;
  end;

  // Check OPEN1 through OPEN20
  for I := 1 to 20 do
  begin
    ValueName := 'OPEN' + IntToStr(I);
    if RegQueryStringValue(HKCU, SubKey, ValueName, Existing) then
    begin
      if Pos(UpperCase('XSTARS.xlam'), UpperCase(Existing)) > 0 then
        Exit;  // already registered
    end
    else
    begin
      // This slot is free — use it
      RegWriteStringValue(HKCU, SubKey, ValueName, '/R "' + XlamPath + '"');
      Exit;
    end;
  end;
end;

// Remove our add-in from OPEN/OPENn values
procedure UnregisterAddinForVersion(Version: String);
var
  SubKey, ValueName, Existing: String;
  I: Integer;
begin
  SubKey := OfficeBase + Version + '\Excel\Options';

  if not RegKeyExists(HKCU, SubKey) then
    Exit;

  // Check OPEN
  if RegQueryStringValue(HKCU, SubKey, 'OPEN', Existing) then
  begin
    if Pos(UpperCase('XSTARS.xlam'), UpperCase(Existing)) > 0 then
      RegDeleteValue(HKCU, SubKey, 'OPEN');
  end;

  // Check OPEN1..OPEN20
  for I := 1 to 20 do
  begin
    ValueName := 'OPEN' + IntToStr(I);
    if RegQueryStringValue(HKCU, SubKey, ValueName, Existing) then
    begin
      if Pos(UpperCase('XSTARS.xlam'), UpperCase(Existing)) > 0 then
        RegDeleteValue(HKCU, SubKey, ValueName);
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  XlamPath: String;
begin
  if CurStep = ssPostInstall then
  begin
    XlamPath := ExpandConstant('{userappdata}') + '\Microsoft\AddIns\XSTARS.xlam';
    RegisterAddinForVersion('14.0', XlamPath);  // Excel 2010
    RegisterAddinForVersion('15.0', XlamPath);  // Excel 2013
    RegisterAddinForVersion('16.0', XlamPath);  // Excel 2016/2019/365
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    UnregisterAddinForVersion('14.0');
    UnregisterAddinForVersion('15.0');
    UnregisterAddinForVersion('16.0');
  end;
end;
