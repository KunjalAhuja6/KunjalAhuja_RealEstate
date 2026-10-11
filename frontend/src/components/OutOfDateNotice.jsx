export default function OutOfDateNotice({ onReRun, isRunning }) {
  return (
    <div className="notice-banner warning-banner" role="status">
      <div className="banner-text">
        <strong>Property details modified:</strong> The property form has been edited since this estimate was generated.
      </div>
      {onReRun && (
        <button
          type="button"
          className="ghost-button btn-small"
          onClick={onReRun}
          disabled={isRunning}
        >
          {isRunning ? "Updating…" : "Update now"}
        </button>
      )}
    </div>
  );
}
