# Triadic Consciousness Engine - Web Visualization

An interactive web application demonstrating the **Triadic Consciousness Engine (TCE)**, a sophisticated cognitive processing system that models consciousness through triadic phase dynamics in support of **Atchley Optimal Dynamics (AOD)** theory.

## Overview

The Triadic Consciousness Engine visualizes how consciousness operates through multiple cognitive phases - analytical, intuitive, and synthesized reasoning. It provides a real-time interface to interact with Claude AI through different cognitive modes, observing neural region activations and cognitive reasoning processes.

## Features

### 🧠 Four Cognitive Phases

1. **Observational Quiescence (Passive)** - Pure receptive consciousness, photonic absorption mode
2. **Apollonian Analysis** - Rational, deductive, crystalline logic engine
3. **Dionysian Synthesis** - Intuitive, holistic, pattern-emergence processor
4. **Hegelian Integration (Synthesis)** - Dialectical fusion, transcendent metacognition

### 🎨 Real-time Visualizations

- **Dynamic Neural Network Canvas** - Animated particle network that adapts to the current cognitive phase
- **System Metrics Dashboard** - Live tracking of consciousness level, cognitive load, energy flow, and integration
- **Neural Region Activation** - Real-time display of five brain regions:
  - Prefrontal Cortex (PFC) - Executive reasoning
  - Temporal-Semantic (TSC) - Language processing
  - Parietal Integration (PIC) - Spatial/mathematical reasoning
  - Limbic Core (LMC) - Emotional processing
  - Hippocampal Memory (HPC) - Memory and pattern matching

### 📊 Cognitive Reasoning Ledger

Detailed logging of each query's processing including:
- Query timestamp and phase
- Neural activation patterns
- Step-by-step thought process
- Performance metrics (processing time, confidence, energy consumed)

### 💬 Interactive Chat Interface

- Multi-turn conversation support with Claude AI
- Phase-specific AI responses shaped by cognitive mode
- Real-time processing visualization
- Comprehensive conversation history

## Installation

### Prerequisites

- Node.js 18+
- npm or yarn
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Setup

1. Navigate to the web directory:
```bash
cd web
```

2. Install dependencies:
```bash
npm install
```

3. Configure your API key:
```bash
cp .env.local.example .env.local
```

4. Edit `.env.local` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=your_actual_api_key_here
```

**Security Note:** The API key is stored server-side only and never exposed to the browser. All Claude API requests are routed through a secure Next.js API route at `/api/chat`.

### Running the Application

#### Development Mode
```bash
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000)

#### Production Build
```bash
npm run build
npm start
```

## Usage

1. **Select a Cognitive Phase** - Click one of the four phase buttons to change the AI's reasoning mode
2. **Enter Your Query** - Type your question or prompt in the text area
3. **Transmit** - Click "TRANSMIT" or press Enter to process the query
4. **Observe** - Watch the neural activations, thought process, and reasoning ledger update in real-time
5. **Review** - Examine the detailed cognitive trace in the reasoning ledger

### Example Queries

Try these queries in different phases to see how the AI's reasoning changes:

**Apollonian (Analytical):**
- "How would you solve a complex optimization problem?"
- "Explain the logical steps to prove a mathematical theorem"

**Dionysian (Intuitive):**
- "What patterns connect consciousness, creativity, and complexity?"
- "Describe the feeling of a breakthrough insight"

**Synthesis (Integration):**
- "How do logic and intuition work together in discovery?"
- "What is the relationship between order and chaos in adaptive systems?"

**Passive (Observational):**
- "What is consciousness?"
- "Tell me about emergence in complex systems"

## Architecture

### Technology Stack

- **Framework:** Next.js 14 (React 18)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **AI:** Claude API (Anthropic)
- **Visualization:** HTML5 Canvas

### Key Components

- `TriadicConsciousnessEngine.tsx` - Main component containing all logic and UI
- `/api/chat/route.ts` - Secure server-side API route for Claude integration
- Phase configuration system with unique properties for each cognitive mode
- Neural region keyword analysis engine
- Real-time canvas animation system
- Conversation history management

### API Architecture

The application uses a secure three-tier architecture:

1. **Client Layer** - React component handles UI and visualization
2. **API Layer** - Next.js API route (`/api/chat`) mediates requests
3. **AI Layer** - Claude API processes queries server-side

**Security Benefits:**
- API key never exposed to browser
- Request validation and error handling
- Rate limiting capability (can be added)
- Audit logging potential

## Connection to AOD Theory

The Triadic Consciousness Engine demonstrates principles from **Atchley Optimal Dynamics**:

1. **Self-Organization** - Neural activations emerge from query analysis, not predetermined rules
2. **Adaptive Dynamics** - The system shifts between cognitive phases based on context
3. **Integration** - Synthesis mode embodies the dialectical integration central to AOD
4. **Resilience** - Multiple reasoning pathways provide cognitive flexibility
5. **Emergence** - Higher-order insights arise from the interaction of different cognitive modes

The triadic structure (Apollonian/Dionysian/Synthesis) mirrors the thesis-antithesis-synthesis dialectic that underlies optimal network dynamics in AOD theory.

## Contributing

This project is part of the Atchley Optimal Dynamics research initiative. For contributions or questions:

- Review the main [AOD README](../README.md)
- Submit issues or pull requests following the project guidelines
- Ensure changes align with AOD theoretical principles

## License

See the [LICENSE](../LICENSE) file in the root directory.

## Author

**Devin Earl Atchley**
Independent Researcher in Complex Systems Theory

## Acknowledgments

- Built with Claude AI (Anthropic)
- Inspired by dialectical philosophy, neuroscience, and complex systems theory
- Part of the broader Atchley Optimal Dynamics theoretical framework

---

**Note:** This is a research prototype demonstrating theoretical concepts. The "neural regions" and "cognitive phases" are conceptual models for exploring consciousness dynamics, not literal brain simulations.
