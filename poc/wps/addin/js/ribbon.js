function OnAddinLoad(ribbonUI) {
    window.Application.ribbonUI = ribbonUI;
    return true;
}

function OnAction(control) {
    if (control.Id !== "xstarsGate0Callback") {
        return true;
    }

    const workbook = window.Application.ActiveWorkbook;
    const workbookName = workbook ? workbook.Name : "（无活动工作簿）";
    const origin = window.location.origin || "（未知）";
    window.alert(
        `XSTARS Gate 0 回调成功\n工作簿：${workbookName}\nOrigin：${origin}`,
    );
    return true;
}

function GetImage() {
    return "images/1.svg";
}

// WPS resolves these Ribbon callbacks by their global names.
window.OnAddinLoad = OnAddinLoad;
window.OnAction = OnAction;
window.GetImage = GetImage;
