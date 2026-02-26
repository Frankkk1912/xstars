Attribute VB_Name = "RibbonCallbacks"
' XSTARS Ribbon Callbacks

Sub RunXSTARS(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run()"
End Sub

Sub RunXSTARSQuick(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_quick()"
End Sub

Sub RunXSTARSWB(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_wb()"
End Sub

Sub RunXSTARSQPCR(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_qpcr()"
End Sub

Sub RunXSTARSCCK8(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_cck8()"
End Sub

Sub RunElisa(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_elisa()"
End Sub

Sub TransformOnly(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_transform_only()"
End Sub

Sub RunStandardCurve(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_standard_curve()"
End Sub

Sub ExportChart(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_export()"
End Sub

Sub ResetSettings(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_reset_settings()"
End Sub

' --- Base Theme ---
Sub SetBaseThemeClassic(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_base_theme('classic')"
End Sub

Sub SetBaseThemeBW(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_base_theme('bw')"
End Sub

Sub SetBaseThemeMinimal(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_base_theme('minimal')"
End Sub

Sub SetBaseThemeDark(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_base_theme('dark')"
End Sub

' --- Journal Preset ---
Sub SetThemeDefault(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_theme('none')"
End Sub

Sub SetThemeNature(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_theme('nature')"
End Sub

Sub SetThemeScience(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_theme('science')"
End Sub

Sub SetThemeCell(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_theme('cell')"
End Sub

Sub SetThemeLancet(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_theme('lancet')"
End Sub

Sub SetThemeNEJM(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_theme('nejm')"
End Sub

Sub SetThemeJAMA(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_theme('jama')"
End Sub

Sub SetThemeBMJ(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_theme('bmj')"
End Sub

' --- Journal Palette ---
Sub SetJPalDefault(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_journal_palette('default')"
End Sub

Sub SetJPalNature(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_journal_palette('nature')"
End Sub

Sub SetJPalScience(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_journal_palette('science')"
End Sub

Sub SetJPalCell(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_journal_palette('cell')"
End Sub

Sub SetJPalLancet(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_journal_palette('lancet')"
End Sub

Sub SetJPalNEJM(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_journal_palette('nejm')"
End Sub

Sub SetJPalJAMA(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_journal_palette('jama')"
End Sub

Sub SetJPalBMJ(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_journal_palette('bmj')"
End Sub

' --- Color Style ---
Sub SetPaletteDefault(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_palette('default')"
End Sub

Sub SetPaletteColorblind(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_palette('colorblind')"
End Sub

Sub SetPaletteVibrant(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_palette('vibrant')"
End Sub

Sub SetPalettePastel(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_palette('pastel')"
End Sub

Sub SetPaletteDeep(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_palette('deep')"
End Sub

Sub SetPaletteMuted(control As IRibbonControl)
    RunPython "import xstars.main; xstars.main.run_set_palette('muted')"
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
