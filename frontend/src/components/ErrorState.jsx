export default function ErrorState({ error, onRetry }) {
  if (!error) return null;
  const message = typeof error === "string" ? error : error.message || "An error occurred.";

  return (
    <div className="state-card error-state" role="alert">
      <h2>Unable to complete request</h2>
      <p className="state-description">{message}</p>
      {onRetry && (
        <button type="button" className="ghost-button btn-small" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
