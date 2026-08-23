export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function seconds(milliseconds: number): string {
  return `${(milliseconds / 1000).toFixed(2)}s`;
}

export function dollars(value: number | null): string {
  return value === null ? "N/A" : `$${value.toFixed(4)}`;
}

export function compactNumber(value: number): string {
  return new Intl.NumberFormat("en-US", { notation: "compact" }).format(value);
}
