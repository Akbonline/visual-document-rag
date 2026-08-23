import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import App from "./App";

afterEach(cleanup);

describe("Evidence Lab", () => {
  it("renders a real Gate C query and its evidence controls", () => {
    render(<App />);

    expect(screen.getAllByText(/How might the observed trends in top-end income inequality/)).toHaveLength(2);
    expect(screen.getByText("Retrieved evidence")).toBeTruthy();
    expect(screen.getAllByText(/ViDoRe V3 HR/)).toHaveLength(2);
  });

  it("switches to the oracle control without a network request", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /Oracle evidence/ }));
    expect(screen.getByText("Controlled answer")).toBeTruthy();
    expect(screen.getByText(/The observed decline in top-end income inequality/)).toBeTruthy();
  });
});
