' register_addin.vbs — Post-install: register XSTARS.xlam as an Excel add-in
' Usage: wscript.exe register_addin.vbs "<full_path_to_xlam>"
'
' This script attaches to a running Excel instance (or starts a hidden one),
' then enables the add-in so it auto-loads on subsequent Excel launches.

Option Explicit

If WScript.Arguments.Count < 1 Then
    WScript.Quit 1
End If

Dim xlamPath
xlamPath = WScript.Arguments(0)

' Check the file exists
Dim fso
Set fso = CreateObject("Scripting.FileSystemObject")
If Not fso.FileExists(xlamPath) Then
    WScript.Quit 2
End If

Dim xl, createdNew
createdNew = False

' Try to attach to an existing Excel instance first
On Error Resume Next
Set xl = GetObject(, "Excel.Application")
If Err.Number <> 0 Then
    Err.Clear
    Set xl = CreateObject("Excel.Application")
    xl.Visible = False
    createdNew = True
End If
On Error GoTo 0

' Register the add-in
On Error Resume Next
Dim addin
Dim found
found = False

' First, try to find it in the existing add-ins list
For Each addin In xl.AddIns
    If LCase(addin.FullName) = LCase(xlamPath) Then
        addin.Installed = True
        found = True
        Exit For
    End If
Next

' If not found, add it via AddIns.Add
If Not found Then
    Dim newAddin
    Set newAddin = xl.AddIns.Add(xlamPath)
    If Not newAddin Is Nothing Then
        newAddin.Installed = True
    End If
End If
On Error GoTo 0

' Only quit Excel if we created it
If createdNew Then
    xl.Quit
End If

Set xl = Nothing
