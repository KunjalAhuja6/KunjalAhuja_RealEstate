export default function LoadingState({ message = "Working on your valuation…" }) {
  return (
    <div className="state-card loading-state" role="status" aria-live="polite">
      <div className="spinner-icon" aria-hidden="true" />
      <h2>{message}</h2>
      <p className="state-description">
        The valuation team is reviewing neighborhood sales, calculating statistical pricing models, and verifying physical property conditions.
      </p>
    </div>
  );
}
