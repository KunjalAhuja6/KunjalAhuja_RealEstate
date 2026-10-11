export default function EmptyState({ title = "Describe a property to value it", message, actionButton }) {
  return (
    <div className="state-card empty-state">
      <h2>{title}</h2>
      <p className="state-description">
        {message ||
          "Fill in the subject property attributes on the left and run a valuation to generate an estimate grounded in market sales and empirical condition checks."}
      </p>
      {actionButton}
    </div>
  );
}
