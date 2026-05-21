/** Curated universe of US-listed symbols available to the palette
 *  and Charts page. The backend's mock watchlist is tiny (5 names),
 *  so the trader can only search inside that set without this list.
 *
 *  When the user picks an off-watchlist symbol the dashboard's
 *  AI panels can't render (no `RecommendationRow`), so the Charts
 *  page falls back to the TradingView embed — which has live data
 *  for every symbol below.
 *
 *  Exchange prefix is what TradingView needs to resolve the chart.
 *  Bare symbols sometimes auto-resolve, but the prefix removes
 *  ambiguity. */

export interface Ticker {
  symbol: string;
  name: string;
  exchange: "NASDAQ" | "NYSE" | "NYSEARCA" | "AMEX" | "CRYPTO";
  sector: string;
}

/** TradingView-format symbol (e.g. "NASDAQ:NVDA"). */
export function tvSymbol(symbol: string): string {
  const t = TICKERS_BY_SYMBOL[symbol.toUpperCase()];
  if (t) return `${t.exchange}:${t.symbol}`;
  return symbol.toUpperCase();
}

/** Look up by symbol — case-insensitive. */
export function lookupTicker(symbol: string): Ticker | undefined {
  return TICKERS_BY_SYMBOL[symbol.toUpperCase()];
}

export const TICKERS: Ticker[] = [
  // Mega-cap tech
  { symbol: "AAPL", name: "Apple", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "MSFT", name: "Microsoft", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "NVDA", name: "NVIDIA", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "GOOGL", name: "Alphabet (A)", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "GOOG", name: "Alphabet (C)", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "AMZN", name: "Amazon", exchange: "NASDAQ", sector: "Consumer Discretionary" },
  { symbol: "META", name: "Meta Platforms", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "TSLA", name: "Tesla", exchange: "NASDAQ", sector: "Consumer Discretionary" },
  { symbol: "AVGO", name: "Broadcom", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "ORCL", name: "Oracle", exchange: "NYSE", sector: "Technology" },
  { symbol: "CRM", name: "Salesforce", exchange: "NYSE", sector: "Technology" },
  { symbol: "ADBE", name: "Adobe", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "AMD", name: "AMD", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "INTC", name: "Intel", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "CSCO", name: "Cisco", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "QCOM", name: "Qualcomm", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "TXN", name: "Texas Instruments", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "IBM", name: "IBM", exchange: "NYSE", sector: "Technology" },
  { symbol: "NOW", name: "ServiceNow", exchange: "NYSE", sector: "Technology" },
  { symbol: "INTU", name: "Intuit", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "PLTR", name: "Palantir", exchange: "NASDAQ", sector: "Technology" },
  { symbol: "SNOW", name: "Snowflake", exchange: "NYSE", sector: "Technology" },
  { symbol: "SHOP", name: "Shopify", exchange: "NYSE", sector: "Technology" },
  { symbol: "UBER", name: "Uber", exchange: "NYSE", sector: "Technology" },
  { symbol: "ABNB", name: "Airbnb", exchange: "NASDAQ", sector: "Consumer Discretionary" },
  { symbol: "NFLX", name: "Netflix", exchange: "NASDAQ", sector: "Communication" },
  { symbol: "DIS", name: "Disney", exchange: "NYSE", sector: "Communication" },

  // Financials
  { symbol: "JPM", name: "JPMorgan Chase", exchange: "NYSE", sector: "Financials" },
  { symbol: "BAC", name: "Bank of America", exchange: "NYSE", sector: "Financials" },
  { symbol: "WFC", name: "Wells Fargo", exchange: "NYSE", sector: "Financials" },
  { symbol: "GS", name: "Goldman Sachs", exchange: "NYSE", sector: "Financials" },
  { symbol: "MS", name: "Morgan Stanley", exchange: "NYSE", sector: "Financials" },
  { symbol: "C", name: "Citigroup", exchange: "NYSE", sector: "Financials" },
  { symbol: "V", name: "Visa", exchange: "NYSE", sector: "Financials" },
  { symbol: "MA", name: "Mastercard", exchange: "NYSE", sector: "Financials" },
  { symbol: "AXP", name: "American Express", exchange: "NYSE", sector: "Financials" },
  { symbol: "BLK", name: "BlackRock", exchange: "NYSE", sector: "Financials" },
  { symbol: "BRK.B", name: "Berkshire Hathaway", exchange: "NYSE", sector: "Financials" },
  { symbol: "SCHW", name: "Charles Schwab", exchange: "NYSE", sector: "Financials" },

  // Healthcare
  { symbol: "UNH", name: "UnitedHealth", exchange: "NYSE", sector: "Healthcare" },
  { symbol: "JNJ", name: "Johnson & Johnson", exchange: "NYSE", sector: "Healthcare" },
  { symbol: "LLY", name: "Eli Lilly", exchange: "NYSE", sector: "Healthcare" },
  { symbol: "PFE", name: "Pfizer", exchange: "NYSE", sector: "Healthcare" },
  { symbol: "MRK", name: "Merck", exchange: "NYSE", sector: "Healthcare" },
  { symbol: "ABBV", name: "AbbVie", exchange: "NYSE", sector: "Healthcare" },
  { symbol: "TMO", name: "Thermo Fisher", exchange: "NYSE", sector: "Healthcare" },
  { symbol: "ABT", name: "Abbott", exchange: "NYSE", sector: "Healthcare" },
  { symbol: "DHR", name: "Danaher", exchange: "NYSE", sector: "Healthcare" },
  { symbol: "BMY", name: "Bristol-Myers", exchange: "NYSE", sector: "Healthcare" },
  { symbol: "AMGN", name: "Amgen", exchange: "NASDAQ", sector: "Healthcare" },
  { symbol: "GILD", name: "Gilead Sciences", exchange: "NASDAQ", sector: "Healthcare" },
  { symbol: "MRNA", name: "Moderna", exchange: "NASDAQ", sector: "Healthcare" },

  // Consumer
  { symbol: "WMT", name: "Walmart", exchange: "NYSE", sector: "Consumer Staples" },
  { symbol: "COST", name: "Costco", exchange: "NASDAQ", sector: "Consumer Staples" },
  { symbol: "PG", name: "Procter & Gamble", exchange: "NYSE", sector: "Consumer Staples" },
  { symbol: "KO", name: "Coca-Cola", exchange: "NYSE", sector: "Consumer Staples" },
  { symbol: "PEP", name: "PepsiCo", exchange: "NASDAQ", sector: "Consumer Staples" },
  { symbol: "MCD", name: "McDonald's", exchange: "NYSE", sector: "Consumer Discretionary" },
  { symbol: "SBUX", name: "Starbucks", exchange: "NASDAQ", sector: "Consumer Discretionary" },
  { symbol: "NKE", name: "Nike", exchange: "NYSE", sector: "Consumer Discretionary" },
  { symbol: "HD", name: "Home Depot", exchange: "NYSE", sector: "Consumer Discretionary" },
  { symbol: "LOW", name: "Lowe's", exchange: "NYSE", sector: "Consumer Discretionary" },
  { symbol: "TGT", name: "Target", exchange: "NYSE", sector: "Consumer Discretionary" },
  { symbol: "CMG", name: "Chipotle", exchange: "NYSE", sector: "Consumer Discretionary" },

  // Industrials & energy
  { symbol: "XOM", name: "ExxonMobil", exchange: "NYSE", sector: "Energy" },
  { symbol: "CVX", name: "Chevron", exchange: "NYSE", sector: "Energy" },
  { symbol: "COP", name: "ConocoPhillips", exchange: "NYSE", sector: "Energy" },
  { symbol: "BA", name: "Boeing", exchange: "NYSE", sector: "Industrials" },
  { symbol: "CAT", name: "Caterpillar", exchange: "NYSE", sector: "Industrials" },
  { symbol: "GE", name: "GE Aerospace", exchange: "NYSE", sector: "Industrials" },
  { symbol: "UPS", name: "UPS", exchange: "NYSE", sector: "Industrials" },
  { symbol: "FDX", name: "FedEx", exchange: "NYSE", sector: "Industrials" },
  { symbol: "LMT", name: "Lockheed Martin", exchange: "NYSE", sector: "Industrials" },
  { symbol: "RTX", name: "RTX", exchange: "NYSE", sector: "Industrials" },
  { symbol: "GM", name: "General Motors", exchange: "NYSE", sector: "Consumer Discretionary" },
  { symbol: "F", name: "Ford", exchange: "NYSE", sector: "Consumer Discretionary" },

  // Communications
  { symbol: "VZ", name: "Verizon", exchange: "NYSE", sector: "Communication" },
  { symbol: "T", name: "AT&T", exchange: "NYSE", sector: "Communication" },
  { symbol: "TMUS", name: "T-Mobile", exchange: "NASDAQ", sector: "Communication" },

  // Popular speculative / meme
  { symbol: "GME", name: "GameStop", exchange: "NYSE", sector: "Consumer Discretionary" },
  { symbol: "AMC", name: "AMC Entertainment", exchange: "NYSE", sector: "Communication" },
  { symbol: "RIVN", name: "Rivian", exchange: "NASDAQ", sector: "Consumer Discretionary" },
  { symbol: "LCID", name: "Lucid Motors", exchange: "NASDAQ", sector: "Consumer Discretionary" },
  { symbol: "COIN", name: "Coinbase", exchange: "NASDAQ", sector: "Financials" },
  { symbol: "HOOD", name: "Robinhood", exchange: "NASDAQ", sector: "Financials" },
  { symbol: "SOFI", name: "SoFi", exchange: "NASDAQ", sector: "Financials" },
  { symbol: "RBLX", name: "Roblox", exchange: "NYSE", sector: "Communication" },
  { symbol: "DKNG", name: "DraftKings", exchange: "NASDAQ", sector: "Consumer Discretionary" },
  { symbol: "U", name: "Unity", exchange: "NYSE", sector: "Technology" },

  // Major ETFs
  { symbol: "SPY", name: "SPDR S&P 500", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "QQQ", name: "Invesco QQQ Trust", exchange: "NASDAQ", sector: "ETF" },
  { symbol: "VOO", name: "Vanguard S&P 500", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "VTI", name: "Vanguard Total Market", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "IWM", name: "iShares Russell 2000", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "DIA", name: "SPDR Dow Jones", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "EFA", name: "iShares MSCI EAFE", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "EEM", name: "iShares MSCI Emerging", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "GLD", name: "SPDR Gold", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "SLV", name: "iShares Silver", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "TLT", name: "iShares 20+Y Treasury", exchange: "NASDAQ", sector: "ETF" },
  { symbol: "HYG", name: "iShares High Yield", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLK", name: "Technology Select", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLF", name: "Financial Select", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLE", name: "Energy Select", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLV", name: "Healthcare Select", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLI", name: "Industrial Select", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLY", name: "Consumer Discretionary", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLP", name: "Consumer Staples", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLB", name: "Materials Select", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLU", name: "Utilities Select", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLRE", name: "Real Estate Select", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "XLC", name: "Communication Services", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "ARKK", name: "ARK Innovation", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "SOXX", name: "iShares Semiconductor", exchange: "NASDAQ", sector: "ETF" },
  { symbol: "SMH", name: "VanEck Semiconductor", exchange: "NASDAQ", sector: "ETF" },
  { symbol: "VIX", name: "CBOE Volatility Index", exchange: "AMEX", sector: "ETF" },
  { symbol: "UVXY", name: "ProShares Ultra VIX", exchange: "NYSEARCA", sector: "ETF" },
  { symbol: "TQQQ", name: "ProShares UltraPro QQQ", exchange: "NASDAQ", sector: "ETF" },
  { symbol: "SQQQ", name: "ProShares UltraPro Short QQQ", exchange: "NASDAQ", sector: "ETF" },

  // Crypto (via TradingView indices)
  { symbol: "BTCUSD", name: "Bitcoin", exchange: "CRYPTO", sector: "Crypto" },
  { symbol: "ETHUSD", name: "Ethereum", exchange: "CRYPTO", sector: "Crypto" },
  { symbol: "SOLUSD", name: "Solana", exchange: "CRYPTO", sector: "Crypto" },
  { symbol: "DOGEUSD", name: "Dogecoin", exchange: "CRYPTO", sector: "Crypto" },
];

const TICKERS_BY_SYMBOL: Record<string, Ticker> = Object.fromEntries(
  TICKERS.map((t) => [t.symbol, t]),
);

export function searchTickers(query: string, limit = 20): Ticker[] {
  const q = query.trim().toUpperCase();
  if (q.length === 0) return TICKERS.slice(0, limit);
  // Exact symbol matches first, then prefix matches on symbol, then
  // substring matches on name. Keeps "AA" → AAPL above ABNB above
  // "Plus AAdvantage".
  const exact: Ticker[] = [];
  const prefix: Ticker[] = [];
  const nameMatch: Ticker[] = [];
  for (const t of TICKERS) {
    if (t.symbol === q) exact.push(t);
    else if (t.symbol.startsWith(q)) prefix.push(t);
    else if (t.name.toUpperCase().includes(q)) nameMatch.push(t);
  }
  return [...exact, ...prefix, ...nameMatch].slice(0, limit);
}
