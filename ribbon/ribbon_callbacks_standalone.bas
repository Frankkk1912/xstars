Attribute VB_Name = "RibbonCallbacks"
' XSTARS Ribbon Callbacks — Standalone (portable) mode
'
' The exe is expected in a "xstars" subfolder next to the workbook.
' Each callback launches xstars.exe via Shell, passing:
'   xstars.exe <command> "<workbook_full_path>"

Private Function ExePath() As String
    ExePath = ThisWorkbook.Path & "\xstars\xstars.exe"
End Function

Private Sub RunCmd(cmd As String)
    Dim exe As String
    Dim wb As String
    exe = ExePath()
    wb = ActiveWorkbook.FullName
    Shell """" & exe & """ " & cmd & " """ & wb & """", vbHide
End Sub

Sub RunXSTARS(control As IRibbonControl)
    RunCmd "run"
End Sub

Sub RunXSTARSQuick(control As IRibbonControl)
    RunCmd "run_quick"
End Sub

Sub RunXSTARSWB(control As IRibbonControl)
    RunCmd "run_wb"
End Sub

Sub RunXSTARSQPCR(control As IRibbonControl)
    RunCmd "run_qpcr"
End Sub

Sub RunXSTARSCCK8(control As IRibbonControl)
    RunCmd "run_cck8"
End Sub

Sub RunElisa(control As IRibbonControl)
    RunCmd "run_elisa"
End Sub

Sub TransformOnly(control As IRibbonControl)
    RunCmd "run_transform_only"
End Sub

Sub RunStandardCurve(control As IRibbonControl)
    RunCmd "run_standard_curve"
End Sub

Sub ExportChart(control As IRibbonControl)
    RunCmd "run_export"
End Sub

Sub ResetSettings(control As IRibbonControl)
    RunCmd "run_reset_settings"
End Sub

' --- Base Theme ---
Sub SetBaseThemeClassic(control As IRibbonControl)
    RunCmd "run_set_base_theme_classic"
End Sub

Sub SetBaseThemeBW(control As IRibbonControl)
    RunCmd "run_set_base_theme_bw"
End Sub

Sub SetBaseThemeMinimal(control As IRibbonControl)
    RunCmd "run_set_base_theme_minimal"
End Sub

Sub SetBaseThemeDark(control As IRibbonControl)
    RunCmd "run_set_base_theme_dark"
End Sub

' --- Journal Preset ---
Sub SetThemeDefault(control As IRibbonControl)
    RunCmd "run_set_theme_none"
End Sub

Sub SetThemeNature(control As IRibbonControl)
    RunCmd "run_set_theme_nature"
End Sub

Sub SetThemeScience(control As IRibbonControl)
    RunCmd "run_set_theme_science"
End Sub

Sub SetThemeCell(control As IRibbonControl)
    RunCmd "run_set_theme_cell"
End Sub

Sub SetThemeLancet(control As IRibbonControl)
    RunCmd "run_set_theme_lancet"
End Sub

Sub SetThemeNEJM(control As IRibbonControl)
    RunCmd "run_set_theme_nejm"
End Sub

Sub SetThemeJAMA(control As IRibbonControl)
    RunCmd "run_set_theme_jama"
End Sub

Sub SetThemeBMJ(control As IRibbonControl)
    RunCmd "run_set_theme_bmj"
End Sub

' --- Journal Palette ---
Sub SetJPalDefault(control As IRibbonControl)
    RunCmd "run_set_journal_palette_default"
End Sub

Sub SetJPalNature(control As IRibbonControl)
    RunCmd "run_set_journal_palette_nature"
End Sub

Sub SetJPalScience(control As IRibbonControl)
    RunCmd "run_set_journal_palette_science"
End Sub

Sub SetJPalCell(control As IRibbonControl)
    RunCmd "run_set_journal_palette_cell"
End Sub

Sub SetJPalLancet(control As IRibbonControl)
    RunCmd "run_set_journal_palette_lancet"
End Sub

Sub SetJPalNEJM(control As IRibbonControl)
    RunCmd "run_set_journal_palette_nejm"
End Sub

Sub SetJPalJAMA(control As IRibbonControl)
    RunCmd "run_set_journal_palette_jama"
End Sub

Sub SetJPalBMJ(control As IRibbonControl)
    RunCmd "run_set_journal_palette_bmj"
End Sub

' --- Color Style ---
Sub SetPaletteDefault(control As IRibbonControl)
    RunCmd "run_set_palette_default"
End Sub

Sub SetPaletteColorblind(control As IRibbonControl)
    RunCmd "run_set_palette_colorblind"
End Sub

Sub SetPaletteVibrant(control As IRibbonControl)
    RunCmd "run_set_palette_vibrant"
End Sub

Sub SetPalettePastel(control As IRibbonControl)
    RunCmd "run_set_palette_pastel"
End Sub

Sub SetPaletteDeep(control As IRibbonControl)
    RunCmd "run_set_palette_deep"
End Sub

Sub SetPaletteMuted(control As IRibbonControl)
    RunCmd "run_set_palette_muted"
End Sub

Sub ShowAbout(control As IRibbonControl)
    Dim msg As String
    msg = "XSTARS v1.0.0" & vbCrLf & vbCrLf & _
          "Quick statistical analysis and publication-quality" & vbCrLf & _
          "visualization inside Excel." & vbCrLf & vbCrLf & _
          "Author: Frank-SYSU" & vbCrLf & _
          "Powered by scipy, matplotlib, seaborn & xlwings." & vbCrLf & vbCrLf & _
          "License: MIT" & vbCrLf & vbCrLf & _
          "Documentation & source:" & vbCrLf & _
          "https://github.com/Frankkk1912/excel-prism"
    MsgBox msg, vbInformation, "About XSTARS"
End Sub
