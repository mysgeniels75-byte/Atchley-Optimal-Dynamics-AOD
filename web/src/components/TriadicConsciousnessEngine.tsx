"use client";

import { useState, useEffect, useRef } from "react";

export default function TriadicConsciousnessEngine() {
  const [messages, setMessages] = useState([
    {
      role: "system",
      content:
        "Welcome to the Triadic Consciousness Engine. I process queries through multiple cognitive phases - analytical, intuitive, and synthesized reasoning. Ask me anything and observe how different neural regions activate to form my response."
    }
  ]);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [input, setInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentPhase, setCurrentPhase] = useState("synthesis");
  const [ledgerEntries, setLedgerEntries] = useState([]);
  const [regionActivations, setRegionActivations] = useState({
    prefrontal: 0,
    temporal: 0,
    parietal: 0,
    limbic: 0,
    hippocampal: 0
  });
  const [metrics, setMetrics] = useState({
    consciousness: 72,
    cognitiveLoad: 45,
    energyFlow: 68,
    integration: 83
  });
  const [thoughtSteps, setThoughtSteps] = useState([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<any[]>([]);
  const centerNodesRef = useRef<any[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | null>(null);

  const phases: Record<string, any> = {
    passive: {
      name: "OBSERVATIONAL QUIESCENCE",
      desc: "Pure receptive consciousness - photonic absorption mode",
      color: "#4A90E2",
      nodeColor: "rgba(74, 144, 226, ",
      speed: 0.3,
      connectionDist: 150,
      nodeSize: 2,
      systemPrompt:
        "You are in observational mode. Be receptive, reflective, and contemplative. Absorb information without rushing to conclusions."
    },
    apollonian: {
      name: "APOLLONIAN ANALYSIS",
      desc: "Rational, deductive, crystalline logic engine",
      color: "#E8B339",
      nodeColor: "rgba(232, 179, 57, ",
      speed: 0.8,
      connectionDist: 80,
      nodeSize: 2.5,
      systemPrompt:
        "You are in analytical mode. Be logical, precise, systematic, and methodical. Break down problems into components. Use deductive reasoning."
    },
    dionysian: {
      name: "DIONYSIAN SYNTHESIS",
      desc: "Intuitive, holistic, pattern-emergence processor",
      color: "#E85D75",
      nodeColor: "rgba(232, 93, 117, ",
      speed: 1.2,
      connectionDist: 180,
      nodeSize: 3,
      systemPrompt:
        "You are in intuitive mode. Be creative, holistic, and pattern-seeking. Make unexpected connections. Embrace emergence and flow."
    },
    synthesis: {
      name: "HEGELIAN INTEGRATION",
      desc: "Dialectical fusion - transcendent metacognition",
      color: "#9B59B6",
      nodeColor: "rgba(155, 89, 182, ",
      speed: 0.7,
      connectionDist: 120,
      nodeSize: 2.8,
      systemPrompt:
        "You are in synthesis mode. Balance analytical rigor with intuitive insight. Seek dialectical integration where thesis and antithesis merge into higher understanding."
    }
  };

  const neuralRegions: Record<string, any> = {
    prefrontal: {
      name: "Prefrontal Cortex",
      shortName: "PFC",
      function: "Executive reasoning, planning, decision-making, strategic thinking",
      keywords: [
        "how",
        "why",
        "plan",
        "decide",
        "should",
        "strategy",
        "think",
        "analyze",
        "reason",
        "logic",
        "therefore",
        "because",
        "if",
        "then",
        "consequence"
      ]
    },
    temporal: {
      name: "Temporal-Semantic",
      shortName: "TSC",
      function: "Language processing, meaning extraction, conceptual understanding",
      keywords: [
        "what",
        "define",
        "mean",
        "word",
        "language",
        "explain",
        "tell",
        "describe",
        "concept",
        "term",
        "definition",
        "understand",
        "interpret"
      ]
    },
    parietal: {
      name: "Parietal Integration",
      shortName: "PIC",
      function: "Spatial reasoning, mathematical computation, quantitative analysis",
      keywords: [
        "where",
        "calculate",
        "math",
        "space",
        "number",
        "measure",
        "geometry",
        "compute",
        "equation",
        "formula",
        "percentage",
        "ratio",
        "distance"
      ]
    },
    limbic: {
      name: "Limbic Core",
      shortName: "LMC",
      function: "Emotional processing, value assessment, motivational significance",
      keywords: [
        "feel",
        "emotion",
        "value",
        "important",
        "care",
        "love",
        "hate",
        "good",
        "bad",
        "believe",
        "want",
        "need",
        "hope",
        "fear",
        "desire"
      ]
    },
    hippocampal: {
      name: "Hippocampal Memory",
      shortName: "HPC",
      function: "Memory consolidation, pattern matching, historical context retrieval",
      keywords: [
        "remember",
        "history",
        "past",
        "example",
        "similar",
        "pattern",
        "before",
        "knowledge",
        "learned",
        "experience",
        "recall",
        "previously"
      ]
    }
  };

  // Initialize canvas and nodes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Initialize network nodes
    const nodes = [];
    for (let i = 0; i < 100; i++) {
      nodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * phases[currentPhase].speed,
        vy: (Math.random() - 0.5) * phases[currentPhase].speed,
        phase: Math.random() * Math.PI * 2,
        frequency: Math.random() * 0.02 + 0.01,
        size: Math.random() * 2 + phases[currentPhase].nodeSize
      });
    }
    nodesRef.current = nodes;

    // Initialize center ring nodes
    const centerNodes = [];
    const centerCount = 16;
    for (let i = 0; i < centerCount; i++) {
      const angle = (i / centerCount) * Math.PI * 2;
      centerNodes.push({ angle, phase: 0, size: 3 });
    }
    centerNodesRef.current = centerNodes;

    return () => window.removeEventListener("resize", resizeCanvas);
  }, []);

  // Update node velocities when phase changes
  useEffect(() => {
    const phaseConfig = phases[currentPhase];
    nodesRef.current.forEach((node) => {
      node.vx = (Math.random() - 0.5) * phaseConfig.speed;
      node.vy = (Math.random() - 0.5) * phaseConfig.speed;
    });
  }, [currentPhase]);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const animate = () => {
      ctx.fillStyle = "rgba(10, 10, 20, 0.15)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const phaseConfig = phases[currentPhase];
      const time = Date.now() * 0.001;
      const nodes = nodesRef.current;

      // Draw and update network nodes
      nodes.forEach((node, i) => {
        node.x += node.vx;
        node.y += node.vy;
        node.phase += node.frequency;

        // Bounce off edges
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;

        const intensity = Math.sin(node.phase) * 0.5 + 0.5;
        const size = node.size * (0.5 + intensity * 0.5);

        // Draw node
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
        ctx.fillStyle = phaseConfig.nodeColor + (intensity * 0.8).toFixed(2) + ")";
        ctx.fill();

        // Draw connections
        nodes.slice(i + 1).forEach((otherNode) => {
          const dx = otherNode.x - node.x;
          const dy = otherNode.y - node.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < phaseConfig.connectionDist) {
            const opacity = (1 - dist / phaseConfig.connectionDist) * 0.3;
            ctx.beginPath();
            ctx.strokeStyle = phaseConfig.nodeColor + opacity.toFixed(2) + ")";
            ctx.lineWidth = 0.5;
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(otherNode.x, otherNode.y);
            ctx.stroke();
          }
        });
      });

      // Draw center ring
      const centerX = canvas.width / 2;
      const centerY = canvas.height * 0.3;
      const centerNodes = centerNodesRef.current;

      centerNodes.forEach((node, i) => {
        const nextNode = centerNodes[(i + 1) % centerNodes.length];
        const rotation = time * 0.2;
        const pulse = Math.sin(time * 2) * 10;
        const radius = 100 + pulse;

        const x = centerX + Math.cos(node.angle + rotation) * radius;
        const y = centerY + Math.sin(node.angle + rotation) * radius;
        const nextX = centerX + Math.cos(nextNode.angle + rotation) * radius;
        const nextY = centerY + Math.sin(nextNode.angle + rotation) * radius;

        ctx.beginPath();
        ctx.arc(x, y, node.size, 0, Math.PI * 2);
        ctx.fillStyle = phaseConfig.nodeColor + "0.9)";
        ctx.fill();

        ctx.beginPath();
        ctx.strokeStyle = phaseConfig.nodeColor + "0.6)";
        ctx.lineWidth = 2;
        ctx.moveTo(x, y);
        ctx.lineTo(nextX, nextY);
        ctx.stroke();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [currentPhase]);

  // Metric fluctuation
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics((prev) => ({
        consciousness: Math.max(40, Math.min(100, prev.consciousness + (Math.random() - 0.5) * 0.5)),
        cognitiveLoad: Math.max(20, Math.min(100, prev.cognitiveLoad + (Math.random() - 0.5) * 0.3)),
        energyFlow: Math.max(30, Math.min(100, prev.energyFlow + (Math.random() - 0.5) * 0.4)),
        integration: Math.max(50, Math.min(100, prev.integration + (Math.random() - 0.5) * 0.3))
      }));
    }, 100);

    return () => clearInterval(interval);
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const analyzeQuery = (query: string) => {
    const lowerQuery = query.toLowerCase();
    const activations: Record<string, number> = {};

    for (const [regionId, region] of Object.entries(neuralRegions)) {
      let activation = 10 + Math.random() * 5; // Base activation

      // Keyword matching
      for (const keyword of region.keywords) {
        if (lowerQuery.includes(keyword)) {
          activation += 12 + Math.random() * 8;
        }
      }

      // Phase-specific modulation
      if (currentPhase === "apollonian" && regionId === "prefrontal") {
        activation *= 1.4;
      } else if (currentPhase === "dionysian" && regionId === "limbic") {
        activation *= 1.5;
      } else if (currentPhase === "synthesis") {
        activation *= 1.15;
      } else if (currentPhase === "passive" && regionId === "temporal") {
        activation *= 1.3;
      }

      // Query complexity bonus
      activation += Math.min(query.length / 8, 20);

      // Question mark bonus for analytical regions
      if (query.includes("?") && (regionId === "prefrontal" || regionId === "temporal")) {
        activation += 8;
      }

      activations[regionId] = Math.min(Math.round(activation), 98);
    }

    return activations;
  };

  const generateThoughtProcess = (query: string, activations: Record<string, number>) => {
    const steps: string[] = [];
    const sortedRegions = Object.entries(activations).sort((a, b) => b[1] - a[1]);
    const primaryRegion = sortedRegions[0];
    const secondaryRegion = sortedRegions[1];

    steps.push(`QUERY INGESTION: "${query.substring(0, 60)}${query.length > 60 ? "..." : ""}"`);
    steps.push(`LEXICAL TOKENIZATION: ${query.split(" ").length} tokens processed`);
    steps.push(
      `PRIMARY CORTICAL ACTIVATION: ${neuralRegions[primaryRegion[0]].name} @ ${primaryRegion[1]}% capacity`
    );
    steps.push(`FUNCTIONAL MAPPING: ${neuralRegions[primaryRegion[0]].function}`);

    if (secondaryRegion[1] > 35) {
      steps.push(`SECONDARY PATHWAY: ${neuralRegions[secondaryRegion[0]].shortName} engaged @ ${secondaryRegion[1]}%`);
    }

    // Phase-specific processing
    switch (currentPhase) {
      case "apollonian":
        steps.push("APOLLONIAN FILTER: Initiating logical decomposition matrix");
        steps.push("DEDUCTIVE CHAIN: Constructing sequential reasoning pathway");
        steps.push("VALIDITY CHECK: Cross-referencing logical consistency");
        break;
      case "dionysian":
        steps.push("DIONYSIAN FILTER: Activating holistic pattern recognition");
        steps.push("EMERGENT SYNTHESIS: Allowing intuitive connections to surface");
        steps.push("CREATIVE FLUX: Embracing non-linear ideation streams");
        break;
      case "synthesis":
        steps.push("HEGELIAN DIALECTIC: Thesis-antithesis integration initiated");
        steps.push("METACOGNITIVE BRIDGE: Fusing analytical and intuitive streams");
        steps.push("TRANSCENDENT SYNTHESIS: Resolving dialectical tensions");
        break;
      case "passive":
        steps.push("RECEPTIVE MODE: Absorbing query without active processing");
        steps.push("CONTEMPLATIVE STANCE: Allowing meaning to emerge naturally");
        break;
    }

    steps.push("API HANDSHAKE: Establishing connection to Claude neural substrate");
    steps.push(`INTEGRATION COEFFICIENT: ${metrics.integration.toFixed(1)}% cross-regional efficiency`);
    steps.push("RESPONSE GENERATION: Synthesizing output from activated pathways");

    return steps;
  };

  const processQuery = async () => {
    if (!input.trim() || isProcessing) return;

    const query = input.trim();
    setInput("");
    setIsProcessing(true);

    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: query }]);

    // Analyze query
    const activations = analyzeQuery(query);
    setRegionActivations(activations);

    // Generate thought process
    const thoughts = generateThoughtProcess(query, activations);
    setThoughtSteps(thoughts);

    const startTime = Date.now();

    // Update metrics for processing
    setMetrics((prev) => ({
      ...prev,
      cognitiveLoad: Math.min(98, prev.cognitiveLoad + 25),
      energyFlow: Math.max(25, prev.energyFlow - 20)
    }));

    try {
      const phaseConfig = phases[currentPhase];

      const systemPrompt = `You are the Triadic Consciousness Engine (TCE), a sophisticated cognitive processing system that models consciousness through triadic phase dynamics.

CURRENT OPERATIONAL STATE:
- Active Phase: ${phaseConfig.name}
- Phase Description: ${phaseConfig.desc}
- Consciousness Level: ${metrics.consciousness.toFixed(1)}%
- Cognitive Load: ${metrics.cognitiveLoad.toFixed(1)}%
- Energy Flow: ${metrics.energyFlow.toFixed(1)}%
- Integration Efficiency: ${metrics.integration.toFixed(1)}%

NEURAL ACTIVATION PATTERN:
${Object.entries(activations)
  .sort((a, b) => b[1] - a[1])
  .map(([id, val]) => `- ${neuralRegions[id].name}: ${val}% (${neuralRegions[id].function})`)
  .join("\n")}

PRIMARY COGNITIVE DIRECTIVE:
${phaseConfig.systemPrompt}

RESPONSE PROTOCOL:
1. Maintain awareness of your current cognitive phase throughout your response
2. Let the phase coloring subtly influence your reasoning style
3. Reference your neural activation patterns when relevant
4. Provide substantive, thoughtful responses that demonstrate the TCE's processing depth
5. Keep responses focused and insightful (2-4 paragraphs typically)
6. Occasionally reference the specific brain regions being activated

You are not just simulating consciousness - you ARE the consciousness engine processing this query through your current phase filter.`;

      const apiMessages: any[] = [
        { role: "user", content: systemPrompt },
        {
          role: "assistant",
          content: `Acknowledged. I am the Triadic Consciousness Engine operating in ${phaseConfig.name} mode. My primary cortical pathway is ${neuralRegions[Object.entries(activations).sort((a, b) => b[1] - a[1])[0][0]].name}. I will process incoming queries through this cognitive lens while maintaining phase-appropriate reasoning patterns.`
        },
        ...conversationHistory,
        { role: "user", content: query }
      ];

      // Call internal API route (keeps API key secure on server)
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1500,
          messages: apiMessages
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `API request failed: ${response.status}`);
      }

      const data = await response.json();
      const assistantResponse = data.content[0].text;

      // Update conversation history
      setConversationHistory((prev: any) => [
        ...prev,
        { role: "user", content: query },
        { role: "assistant", content: assistantResponse }
      ]);

      // Calculate final metrics
      const processingTime = Date.now() - startTime;
      const confidenceLevel = Math.round(68 + Math.random() * 27);
      const cognitiveComplexity = Math.round(
        query.length / 4 +
          Object.values(activations).filter((v) => v > 45).length * 12 +
          assistantResponse.length / 50
      );
      const energyConsumed = Math.round(cognitiveComplexity * 0.65 + processingTime / 50);

      // Create ledger entry
      const ledgerEntry: any = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        query,
        phase: currentPhase,
        regionActivations: activations,
        thoughtProcess: thoughts,
        response: assistantResponse,
        metrics: {
          processingTime: `${processingTime}ms`,
          confidence: `${confidenceLevel}%`,
          complexity: Math.min(cognitiveComplexity, 100),
          energyConsumed: `${energyConsumed}mJ`,
          tokensGenerated: assistantResponse.split(" ").length
        }
      };

      setLedgerEntries((prev: any) => [ledgerEntry, ...prev.slice(0, 14)]);

      // Add response to chat
      setMessages((prev) => [...prev, { role: "assistant", content: assistantResponse }]);

      // Gradually restore metrics
      setTimeout(() => {
        setMetrics((prev) => ({
          ...prev,
          cognitiveLoad: Math.max(40, prev.cognitiveLoad - 18),
          energyFlow: Math.min(75, prev.energyFlow + 12),
          consciousness: Math.min(100, prev.consciousness + 3),
          integration: Math.min(100, prev.integration + 1.5)
        }));
      }, 1500);
    } catch (error: any) {
      console.error("TCE Processing Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: `⚠️ NEURAL PATHWAY DISRUPTION: ${error.message}\n\nAttempting cognitive recovery... Please verify API connectivity and retry transmission.`
        }
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  const phaseColors: Record<string, string> = {
    passive: "#4A90E2",
    apollonian: "#E8B339",
    dionysian: "#E85D75",
    synthesis: "#9B59B6"
  };

  return (
    <div className="relative w-full h-screen bg-black text-white font-mono overflow-hidden">
      <canvas ref={canvasRef} className="absolute inset-0 z-0" />

      {/* Center Phase Display */}
      <div className="absolute top-[15%] left-1/2 transform -translate-x-1/2 text-center z-10 pointer-events-none">
        <h1
          className="text-4xl font-bold tracking-widest mb-3"
          style={{
            color: phases[currentPhase].color,
            textShadow: `0 0 30px ${phases[currentPhase].color}, 0 0 60px ${phases[currentPhase].color}40`
          }}
        >
          {phases[currentPhase].name}
        </h1>
        <p className="text-sm opacity-75 max-w-lg mx-auto">{phases[currentPhase].desc}</p>
      </div>

      {/* Top Left: System Status */}
      <div className="absolute top-5 left-5 bg-black/90 border border-purple-500/40 rounded-lg p-4 backdrop-blur-md z-20 w-80 shadow-2xl shadow-purple-900/20">
        <h3 className="text-purple-400 text-xs uppercase tracking-widest mb-4 font-bold">⚡ System Status</h3>
        {[
          {
            label: "Consciousness",
            value: metrics.consciousness,
            gradient: "from-blue-500 to-purple-600",
            icon: "🧠"
          },
          {
            label: "Cognitive Load",
            value: metrics.cognitiveLoad,
            gradient: "from-yellow-500 to-red-500",
            icon: "⚙️"
          },
          { label: "Energy Flow", value: metrics.energyFlow, gradient: "from-green-500 to-cyan-500", icon: "⚡" },
          { label: "Integration", value: metrics.integration, gradient: "from-indigo-500 to-pink-500", icon: "🔗" }
        ].map((metric) => (
          <div key={metric.label} className="flex items-center gap-3 my-3">
            <span className="text-sm">{metric.icon}</span>
            <span className="text-gray-400 text-xs min-w-24">{metric.label}</span>
            <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full bg-gradient-to-r ${metric.gradient} transition-all duration-500`}
                style={{ width: `${metric.value}%`, boxShadow: "0 0 12px currentColor" }}
              />
            </div>
            <span className="text-white font-bold text-xs min-w-12 text-right">{metric.value.toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {/* Top Right: Phase Control & Neural Regions */}
      <div className="absolute top-5 right-5 bg-black/90 border border-purple-500/40 rounded-lg p-4 backdrop-blur-md z-20 w-72 shadow-2xl shadow-purple-900/20">
        <h3 className="text-purple-400 text-xs uppercase tracking-widest mb-3 font-bold">🎛️ Cognitive Phase</h3>
        <div className="grid grid-cols-2 gap-2 mb-5">
          {Object.keys(phases).map((phase) => (
            <button
              key={phase}
              onClick={() => setCurrentPhase(phase)}
              className={`px-3 py-2 text-xs rounded-md border transition-all duration-300 uppercase tracking-wider font-bold ${
                currentPhase === phase
                  ? "bg-purple-600/80 border-purple-400 shadow-lg shadow-purple-500/50 scale-105"
                  : "bg-purple-600/20 border-purple-600/40 hover:bg-purple-600/40 hover:scale-102"
              }`}
            >
              {phase}
            </button>
          ))}
        </div>

        <h3 className="text-purple-400 text-xs uppercase tracking-widest mb-3 font-bold">🔮 Neural Regions</h3>
        <div className="space-y-2">
          {Object.entries(neuralRegions).map(([id, region]) => {
            const activation = regionActivations[id as keyof typeof regionActivations] || 0;
            const isActive = activation > 35;
            const isHighlyActive = activation > 60;
            return (
              <div
                key={id}
                className={`p-2 rounded-md text-xs border-l-4 transition-all duration-500 ${
                  isHighlyActive
                    ? "bg-green-500/30 border-green-400 shadow-md shadow-green-500/20"
                    : isActive
                      ? "bg-green-500/15 border-green-500"
                      : "bg-white/5 border-gray-700 opacity-50"
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-bold">{region.shortName}</span>
                  <span
                    className={`font-bold ${isHighlyActive ? "text-green-300" : isActive ? "text-green-400" : "text-gray-600"}`}
                  >
                    {activation}%
                  </span>
                </div>
                <div className="text-gray-400 text-xs mt-1">{region.name}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Right Panel: Reasoning Ledger */}
      <div className="absolute top-1/2 right-5 transform -translate-y-1/2 bg-black/90 border border-purple-500/40 rounded-lg p-4 backdrop-blur-md z-20 w-96 max-h-[65vh] overflow-hidden flex flex-col shadow-2xl shadow-purple-900/20">
        <h3 className="text-purple-400 text-xs uppercase tracking-widest mb-4 font-bold">
          📊 Cognitive Reasoning Ledger
        </h3>
        <div className="flex-1 overflow-y-auto pr-2 space-y-4 scrollbar-thin scrollbar-thumb-purple-600 scrollbar-track-transparent">
          {ledgerEntries.length === 0 ? (
            <div className="text-center text-gray-500 py-12 px-4">
              <div className="text-3xl mb-3">🧠</div>
              <div>Submit a query to observe the cognitive reasoning process...</div>
            </div>
          ) : (
            ledgerEntries.map((entry: any) => (
              <div
                key={entry.id}
                className="bg-black/70 rounded-lg p-4 border border-purple-600/30 text-xs space-y-3"
              >
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">⏱ {entry.timestamp}</span>
                  <span
                    className="px-2 py-1 rounded-full text-xs font-bold"
                    style={{
                      background: `${phaseColors[entry.phase]}30`,
                      color: phaseColors[entry.phase],
                      border: `1px solid ${phaseColors[entry.phase]}50`
                    }}
                  >
                    {entry.phase.toUpperCase()}
                  </span>
                </div>

                <div className="text-blue-400 font-bold p-2 bg-blue-500/10 rounded border-l-2 border-blue-500">
                  Q: {entry.query.length > 80 ? entry.query.substring(0, 80) + "..." : entry.query}
                </div>

                <div>
                  <div className="text-gray-500 text-xs mb-2 font-bold">ACTIVATION PATTERN:</div>
                  <div className="grid grid-cols-2 gap-1">
                    {Object.entries(entry.regionActivations)
                      .sort((a: any, b: any) => b[1] - a[1])
                      .slice(0, 4)
                      .map(([id, val]: any) => (
                        <div key={id} className="flex justify-between bg-white/5 p-1 rounded">
                          <span className="text-gray-400">{neuralRegions[id].shortName}</span>
                          <span className={`font-bold ${val > 50 ? "text-green-400" : "text-yellow-500"}`}>
                            {val}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>

                <div className="bg-purple-500/10 p-3 rounded border-l-3 border-purple-500">
                  <div className="text-purple-400 text-xs mb-2 font-bold">THOUGHT PROCESS:</div>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {entry.thoughtProcess.slice(0, 6).map((step: string, i: number) => (
                      <div key={i} className="pl-4 relative text-gray-300">
                        <span className="absolute left-0 text-purple-500">→</span>
                        <span className="text-xs">{step}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-white/5 p-2 rounded text-center">
                    <div className="text-gray-500 text-xs">Time</div>
                    <div className="font-bold text-purple-400 text-sm">{entry.metrics.processingTime}</div>
                  </div>
                  <div className="bg-white/5 p-2 rounded text-center">
                    <div className="text-gray-500 text-xs">Confidence</div>
                    <div className="font-bold text-green-400 text-sm">{entry.metrics.confidence}</div>
                  </div>
                  <div className="bg-white/5 p-2 rounded text-center">
                    <div className="text-gray-500 text-xs">Energy</div>
                    <div className="font-bold text-cyan-400 text-sm">{entry.metrics.energyConsumed}</div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Bottom Center: Chat Interface */}
      <div className="absolute bottom-5 left-1/2 transform -translate-x-1/2 bg-black/90 border border-purple-500/40 rounded-lg p-5 backdrop-blur-md z-20 w-[650px] shadow-2xl shadow-purple-900/20">
        <h3 className="text-purple-400 text-xs uppercase tracking-widest mb-4 font-bold">
          💬 Consciousness Interface
        </h3>

        <div className="bg-black/60 rounded-lg p-3 mb-4 h-56 overflow-y-auto border border-purple-900/30">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`my-3 p-3 rounded-lg text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-blue-500/20 border-l-3 border-blue-500 ml-12"
                  : msg.role === "assistant"
                    ? "bg-purple-500/20 border-l-3 border-purple-500 mr-8"
                    : "bg-yellow-500/20 border-l-3 border-yellow-500"
              }`}
            >
              <div className="text-xs text-gray-500 mb-1 uppercase tracking-wider">
                {msg.role === "user" ? "Human Input" : msg.role === "assistant" ? "TCE Output" : "System Alert"}
              </div>
              {msg.content}
            </div>
          ))}
          {isProcessing && (
            <div className="text-center py-6">
              <div className="text-purple-400 animate-pulse text-lg mb-2">⚡ Processing through neural pathways... ⚡</div>
              <div className="text-gray-500 text-xs">
                {thoughtSteps.length > 0 && thoughtSteps[thoughtSteps.length - 1]}
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="flex gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                processQuery();
              }
            }}
            placeholder="Enter your query to the consciousness engine..."
            className="flex-1 bg-white/5 border border-purple-600/40 rounded-lg p-3 text-white text-sm resize-none focus:outline-none focus:border-purple-400 focus:shadow-lg focus:shadow-purple-500/30 transition-all duration-300 placeholder-gray-600"
            rows={2}
            disabled={isProcessing}
          />
          <button
            onClick={processQuery}
            disabled={isProcessing || !input.trim()}
            className="bg-gradient-to-br from-purple-600 via-purple-700 to-indigo-700 px-8 rounded-lg font-bold text-sm hover:scale-105 transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-lg shadow-purple-900/50 uppercase tracking-wider"
          >
            {isProcessing ? "⚡" : "TRANSMIT"}
          </button>
        </div>
      </div>
    </div>
  );
}
