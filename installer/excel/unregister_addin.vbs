' unregister_addin.vbs — Pre-uninstall: disable XSTARS add-in in Excel
' Usage: wscript.exe unregister_addin.vbs "<full_path_to_xlam>"

Option Explicit

If WScript.Arguments.Count < 1 Then
    WScript.Quit 1
End If

Dim xlamPath
xlamPath = WScript.Arguments(0)

Dim xl, createdNew
createdNew = False

On Error Resume Next
Set xl = GetObject(, "Excel.Application")
If Err.Number <> 0 Then
    Err.Clear
    Set xl = CreateObject("Excel.Application")
    xl.Visible = False
    createdNew = True
End If
On Error GoTo 0

' Disable the add-in
On Error Resume Next
Dim addin
For Each addin In xl.AddIns
    If LCase(addin.FullName) = LCase(xlamPath) Then
        addin.Installed = False
        Exit For
    End If
Next
On Error GoTo 0

If createdNew Then
    xl.Quit
End If

Set xl = Nothing
