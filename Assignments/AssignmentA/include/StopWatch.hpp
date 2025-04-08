// StopWatch.hpp
// 
// Simple timer to record duration
// 

#pragma once

#include <chrono>
#include <iostream>

// StopWatch where start and stop can be manually set
class StopWatch
{
public:
    StopWatch();
	explicit StopWatch(const std::string& name);
	StopWatch(const StopWatch&) = delete;
	StopWatch& operator=(const StopWatch&) = delete;

	void Start();
	void Stop();
	void Reset();
    void set_name(const std::string& name);

	double GetTime() const;
    void display_time() const;

private:
    std::string m_name;
	std::chrono::time_point<std::chrono::high_resolution_clock> m_startTimePoint;
	std::chrono::time_point<std::chrono::high_resolution_clock> m_endTimePoint;
	double m_duration;
	bool m_isRunning = false;
};

// Timer 
#include <iostream>
#include <chrono>
#include <string_view>

class Timer {
public:
    Timer() { Reset(); }
    
    void Reset() { m_Start = std::chrono::high_resolution_clock::now(); }
    
    // Elapsed time in seconds
    float Elapsed() const {
        return std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - m_Start).count();
    }

    // Elapsed time in milliseconds
    float ElapsedMillis() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_Start).count();
    }

    // Elapsed time in microseconds
    float ElapsedMicros() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - m_Start).count();
    }

    // Elapsed time in nanoseconds
    float ElapsedNanos() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - m_Start).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
};

class ScopedTimer {
public:
    ScopedTimer(std::string_view name) : m_Name(name) {}
    
    ~ScopedTimer() {
        float time = m_Timer.Elapsed(); // Elapsed time in nanoseconds
        std::cout << m_Name << " - " << time << "s\n";
    }

private:
    Timer m_Timer;
    std::string m_Name;
};

using Clock = std::chrono::high_resolution_clock;

StopWatch::StopWatch()
	: m_name{}, m_startTimePoint{ Clock::now() }, m_endTimePoint{ m_startTimePoint }, m_duration{}, m_isRunning{ false }
{}

StopWatch::StopWatch(const std::string& name)
	: m_name{name}, m_startTimePoint{ Clock::now() }, m_endTimePoint{ m_startTimePoint }, m_duration{}, m_isRunning{ false }
{}

void StopWatch::Start()
{
	if (!m_isRunning)
	{
		m_startTimePoint = Clock::now();
		m_isRunning = true;
	}
	// Does nothing if stopwatch is already running
}

void StopWatch::set_name(const std::string& name)
{
	m_name = name;
}

void StopWatch::Stop()
{
	if (m_isRunning)
	{
		m_endTimePoint = Clock::now();
		std::chrono::duration<double> elapsed_time = m_endTimePoint - m_startTimePoint;
		m_duration += elapsed_time.count();
		m_isRunning = false;
	}
	// Does nothing if stopwatch is not running
}

void StopWatch::Reset()
{
	m_startTimePoint = Clock::now();
	m_endTimePoint = m_startTimePoint;
	m_duration = 0.0;
}

double StopWatch::GetTime() const
{
	if (m_isRunning)
	{
		std::chrono::duration<double> elapsed_time = Clock::now() - m_startTimePoint;
		return elapsed_time.count() + m_duration;
	}

	return m_duration;
}

void StopWatch::display_time() const
{
	std::cout << m_name << " - " << m_duration << "s\n";
}