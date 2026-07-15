import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine 
} from 'recharts';
import { Calendar, AlertTriangle, TrendingUp, ShieldAlert, Activity } from 'lucide-react';

const API_BASE = "http://127.0.0.1:5000/api/v1";

function App() {
  const [prices, setPrices] = useState([]);
  const [shocks, setShocks] = useState([]);
  const [bayesianData, setBayesianData] = useState(null);
  const [selectedShock, setSelectedShock] = useState(null);
  const [startDate, setStartDate] = useState("2010-01-01");
  const [endDate, setEndDate] = useState("2022-09-30");
  const [loading, setLoading] = useState(true);

  // Fetch data from local Flask server when dates change
  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API_BASE}/prices?start_date=${startDate}&end_date=${endDate}`).then(r => r.json()),
      fetch(`${API_BASE}/shocks`).then(r => r.json()),
      fetch(`${API_BASE}/change_points`).then(r => r.json())
    ])
    .then(([pricesRes, shocksRes, bayesianRes]) => {
      if (pricesRes.status === "success") setPrices(pricesRes.data);
      if (shocksRes.status === "success") setShocks(shocksRes.data);
      if (bayesianRes.status === "success") setBayesianData(bayesianRes.data);
      setLoading(false);
    })
    .catch(err => {
      console.error("Failed to load dashboard backend data:", err);
      setLoading(false);
    });
  }, [startDate, endDate]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans">
      {/* Header Banner */}
      <header className="border-b border-gray-800 bg-gray-950 px-6 py-4 flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Activity className="text-yellow-500 animate-pulse" />
            Birhan Energies • Brent Crude Intelligence Dashboard
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            Analyzing macroeconomic and geopolitical regime shifts with Bayesian MCMC inference.
          </p>
        </div>

        {/* Date Filters Control Panel */}
        <div className="flex flex-wrap items-center gap-3 bg-gray-900 p-2 rounded-lg border border-gray-800">
          <div className="flex items-center gap-2">
            <Calendar size={16} className="text-gray-400" />
            <input 
              type="date" 
              value={startDate} 
              onChange={(e) => setStartDate(e.target.value)}
              className="bg-gray-800 text-sm text-gray-200 border-none rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-yellow-500"
            />
          </div>
          <span className="text-gray-600">to</span>
          <input 
            type="date" 
            value={endDate} 
            onChange={(e) => setEndDate(e.target.value)}
            className="bg-gray-800 text-sm text-gray-200 border-none rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-yellow-500"
          />
        </div>
      </header>

      {loading ? (
        <div className="flex items-center justify-center h-[70vh]">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-yellow-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-gray-400">Loading analytical models and data assets...</p>
          </div>
        </div>
      ) : (
        <main className="p-6 max-w-[1600px] mx-auto space-y-6">
          {/* Top Row: Bayesian Inference Metrics & State Cards */}
          <section className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-950 p-5 rounded-xl border border-gray-800">
              <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">Bayesian Change-Point</span>
              <p className="text-xl font-bold text-yellow-500 mt-2">
                {bayesianData ? bayesianData.calculated_break_date : "Pending Calculation"}
              </p>
              <p className="text-xs text-gray-400 mt-1">Identified Structural Breakpoint</p>
            </div>

            <div className="bg-gray-950 p-5 rounded-xl border border-gray-800">
              <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">Model Precision (95% HPDI)</span>
              <p className="text-xl font-bold text-white mt-2">
                {bayesianData ? `[${bayesianData.hpdi_95_lower_date} to ${bayesianData.hpdi_95_upper_date}]` : "N/A"}
              </p>
              <p className="text-xs text-gray-400 mt-1">Highest Posterior Density Interval</p>
            </div>

            <div className="bg-gray-950 p-5 rounded-xl border border-gray-800">
              <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">Pre-Break Volatility (σ₁)</span>
              <p className="text-xl font-bold text-green-400 mt-2">
                {bayesianData ? `${(bayesianData.parameters_before_break.sigma_1_volatility * 100).toFixed(3)}%` : "N/A"}
              </p>
              <p className="text-xs text-gray-400 mt-1">Daily Standard Deviation (Regime 1)</p>
            </div>

            <div className="bg-gray-950 p-5 rounded-xl border border-gray-800">
              <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">Post-Break Volatility (σ₂)</span>
              <p className="text-xl font-bold text-red-400 mt-2">
                {bayesianData ? `${(bayesianData.parameters_after_break.sigma_2_volatility * 100).toFixed(3)}%` : "N/A"}
              </p>
              <p className="text-xs text-gray-400 mt-1">Daily Standard Deviation (Regime 2)</p>
            </div>
          </section>

          {/* Middle Row: Main Chart & Side Details */}
          <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Interactive Timeline Chart (Recharts) */}
            <div className="lg:col-span-2 bg-gray-950 p-6 rounded-xl border border-gray-800">
              <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-white">
                <TrendingUp size={20} className="text-yellow-500" />
                Brent Spot Price ($) & Dynamic Shock Vector Overlay
              </h3>
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={prices} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="Date" stroke="#9ca3af" fontSize={12} tickLine={false} />
                    <YAxis stroke="#9ca3af" fontSize={12} tickLine={false} />
                    <Tooltip contentStyle={{ backgroundColor: '#030712', borderColor: '#374151', color: '#f3f4f6' }} />
                    <Legend />
                    <Line type="monotone" dataKey="Price" stroke="#eab308" strokeWidth={2} dot={false} activeDot={{ r: 6 }} name="Spot Price (USD)" />
                    
                    {/* Visual Overlay of the identified Change Point */}
                    {bayesianData && (
                      <ReferenceLine 
                        x={bayesianData.calculated_break_date} 
                        stroke="#ef4444" 
                        strokeDasharray="5 5" 
                        label={{ value: 'Bayesian Break', fill: '#ef4444', position: 'top', fontSize: 11 }} 
                      />
                    )}

                    {/* Interactive Geopolitical Event Overlays */}
                    {shocks.map((event, idx) => (
                      <ReferenceLine 
                        key={idx} 
                        x={event.Date} 
                        stroke="#3b82f6" 
                        strokeWidth={1}
                        strokeDasharray="3 3"
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Right Panel: Selected Geopolitical Event Details */}
            <div className="bg-gray-950 p-6 rounded-xl border border-gray-800 flex flex-col justify-between">
              <div>
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-white">
                  <ShieldAlert size={20} className="text-blue-500" />
                  Geopolitical Events Panel
                </h3>
                <p className="text-xs text-gray-400 mb-4">
                  Select a registered shock event to inspect underlying macroeconomic causes and market regime shifts.
                </p>

                {/* Vertical Scrollable Events List */}
                <div className="space-y-2 max-h-[250px] overflow-y-auto pr-2 custom-scrollbar">
                  {shocks.map((event, idx) => (
                    <button 
                      key={idx}
                      onClick={() => setSelectedShock(event)}
                      className={`w-full text-left p-3 rounded-lg border transition-all text-sm flex justify-between items-center ${
                        selectedShock?.Event === event.Event 
                          ? 'bg-blue-950/40 border-blue-500 text-blue-200' 
                          : 'bg-gray-900 border-gray-800 hover:border-gray-700 text-gray-300'
                      }`}
                    >
                      <span className="truncate pr-2 font-medium">{event.Event}</span>
                      <span className="text-xs text-gray-500 whitespace-nowrap">{event.Date}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Event Detailed Card */}
              {selectedShock ? (
                <div className="mt-6 p-4 bg-gray-900 border border-blue-900/50 rounded-xl">
                  <div className="flex items-center gap-2 text-blue-400 font-semibold text-sm mb-2">
                    <AlertTriangle size={16} />
                    {selectedShock.Date}
                  </div>
                  <h4 className="font-bold text-white mb-2">{selectedShock.Event}</h4>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    {selectedShock.Description || "Historical global disruption event mapping to shifts in crude baseline pricing structures and OPEC distribution quotas."}
                  </p>
                </div>
              ) : (
                <div className="mt-6 p-4 border border-dashed border-gray-800 rounded-xl text-center text-gray-500 text-xs">
                  Click an event from the list to display structural details and model overlays.
                </div>
              )}
            </div>
          </section>
        </main>
      )}
    </div>
  );
}

export default App;