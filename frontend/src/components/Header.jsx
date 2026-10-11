export default function Header({ statusText, isOffline, theme, onToggleTheme }) {
  return (
    <header className="app-header">
      <div className="header-brand">
        <h1>Property Valuation Workbench</h1>
        <span className={`status-badge ${isOffline ? "offline" : ""}`}>
          {statusText}
        </span>
      </div>
      <button
        type="button"
        className="theme-toggle"
        onClick={onToggleTheme}
        aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      >
        {theme === "dark" ? "Light mode" : "Dark mode"}
      </button>
    </header>
  );
}
