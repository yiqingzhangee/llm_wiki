(function () {
  var rootStyle = getComputedStyle(document.documentElement);
  var colors = {
    bg: rootStyle.getPropertyValue("--bg").trim(),
    bg2: rootStyle.getPropertyValue("--bg2").trim(),
    ink: rootStyle.getPropertyValue("--ink").trim(),
    muted: rootStyle.getPropertyValue("--muted").trim(),
    rule: rootStyle.getPropertyValue("--rule").trim(),
    accent: rootStyle.getPropertyValue("--accent").trim(),
    accentSoft: rootStyle.getPropertyValue("--accent-soft").trim(),
    accent2: rootStyle.getPropertyValue("--accent2").trim(),
    accent2Soft: rootStyle.getPropertyValue("--accent2-soft").trim()
  };

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
    theme: "base",
    flowchart: {
      curve: "linear",
      htmlLabels: true,
      nodeSpacing: 22,
      rankSpacing: 28,
      useMaxWidth: true
    },
    themeVariables: {
      fontFamily: "Instrument",
      fontSize: "14px",
      primaryColor: colors.bg2,
      primaryTextColor: colors.ink,
      primaryBorderColor: colors.rule,
      lineColor: colors.muted,
      secondaryColor: colors.accentSoft,
      tertiaryColor: colors.accent2Soft,
      clusterBkg: colors.bg,
      clusterBorder: colors.rule,
      edgeLabelBackground: colors.bg2
    }
  });

  mermaid.run({ querySelector: ".mermaid" });

  var buttons = document.querySelectorAll("[data-focus-button]");
  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      var focus = button.getAttribute("data-focus-button");
      document.body.setAttribute("data-focus", focus);
      buttons.forEach(function (item) {
        item.setAttribute("aria-pressed", item === button ? "true" : "false");
      });
    });
  });
})();
