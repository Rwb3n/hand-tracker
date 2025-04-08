"""
Contains filter classes for smoothing cursor movements (e.g., Kalman, Moving Average).
"""

import numpy as np

class KalmanFilter2D:
    """A simple Kalman filter implementation for 2D point smoothing.

    Assumes a constant velocity model.
    State vector: [x, y, vx, vy]
    Measurement vector: [x, y]
    """
    def __init__(self, dt=0.1, process_noise_std=1e-2, measurement_noise_std=1e-1):
        """Initializes the Kalman filter.

        Args:
            dt (float): Time step between measurements.
            process_noise_std (float): Standard deviation of process noise.
            measurement_noise_std (float): Standard deviation of measurement noise.
        """
        self.dt = dt

        # State transition matrix (A)
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix (H)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance (Q)
        # Simplified: assuming independent noise in position and velocity components
        q_pos = process_noise_std**2
        q_vel = (process_noise_std * dt)**2 # Scale noise by dt for velocity
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        # Measurement noise covariance (R)
        self.R = np.diag([measurement_noise_std**2, measurement_noise_std**2])

        # Initial state estimate ([x, y, vx, vy]) - start at origin with zero velocity
        self.x_hat = np.zeros((4, 1))

        # Initial estimate covariance (P)
        self.P = np.eye(4) * 1 # Start with some uncertainty

        self.is_initialized = False

    def predict(self):
        """Predicts the next state."""
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_hat[:2].flatten() # Return predicted position [x, y]

    def update(self, measurement):
        """Updates the state estimate based on a new measurement.

        Args:
            measurement (np.ndarray or list/tuple): The measured [x, y] position.

        Returns:
            np.ndarray: The updated [x, y] position estimate.
        """
        if not self.is_initialized:
            # Initialize state with the first measurement
            self.x_hat[0, 0] = measurement[0]
            self.x_hat[1, 0] = measurement[1]
            self.P = np.eye(4) * 1 # Reset uncertainty
            self.is_initialized = True
            return np.array(measurement)

        measurement = np.array(measurement).reshape(2, 1)

        # Kalman gain (K)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update estimate with measurement (z)
        y = measurement - self.H @ self.x_hat # Measurement residual
        self.x_hat = self.x_hat + K @ y

        # Update estimate covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        return self.x_hat[:2].flatten() # Return updated position [x, y]

    def filter(self, measurement):
        """Convenience method to perform predict and update in one step."""
        self.predict()
        return self.update(measurement)


# Example usage (for testing this module directly)
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("Running Kalman Filter Test...")
    # Simulate noisy measurements (e.g., a slightly wobbly line)
    true_x = np.linspace(0, 10, 100)
    true_y = 0.5 * true_x + 2
    measurements_x = true_x + np.random.normal(0, 0.5, 100)
    measurements_y = true_y + np.random.normal(0, 0.5, 100)
    measurements = list(zip(measurements_x, measurements_y))

    # Initialize filter
    kalman = KalmanFilter2D(dt=0.1, process_noise_std=0.1, measurement_noise_std=0.5)
    kalman_filtered_points = []

    # Apply filter
    for meas in measurements:
        predicted = kalman.predict()
        filtered = kalman.update(meas)
        kalman_filtered_points.append(filtered)

    kalman_filtered_points = np.array(kalman_filtered_points)

    # Plotting Kalman results
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(true_x, true_y, 'g-', label='True Path', linewidth=2)
    plt.scatter(measurements_x, measurements_y, c='r', marker='.', label='Noisy Measurements')
    plt.plot(kalman_filtered_points[:, 0], kalman_filtered_points[:, 1], 'b-', label='Kalman Filtered Path', linewidth=2)
    plt.title('Kalman Filter 2D Example')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    print("Kalman filter test complete.")


class MovingAverageFilter2D:
    """A simple moving average filter for 2D points."""
    def __init__(self, window_size=5):
        """Initializes the moving average filter.

        Args:
            window_size (int): The number of past points to average.
        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1.")
        self.window_size = window_size
        self.points = [] # Store recent points as (x, y) tuples

    def filter(self, point):
        """Adds a new point and returns the smoothed average.

        Args:
            point (list or tuple): The new (x, y) point.

        Returns:
            tuple: The smoothed (x, y) point.
        """
        self.points.append(point)

        # Keep only the last `window_size` points
        if len(self.points) > self.window_size:
            self.points.pop(0)

        # Calculate the average
        if not self.points:
            return (0.0, 0.0) # Should not happen if called correctly

        avg_x = sum(p[0] for p in self.points) / len(self.points)
        avg_y = sum(p[1] for p in self.points) / len(self.points)

        return (avg_x, avg_y)



if __name__ == '__main__':
    # Moving Average Test
    print("\nRunning Moving Average Filter Test...")
    ma_filter = MovingAverageFilter2D(window_size=10)
    ma_filtered_points = []

    for meas in measurements:
        filtered = ma_filter.filter(meas)
        ma_filtered_points.append(filtered)

    ma_filtered_points = np.array(ma_filtered_points)

    # Plotting Moving Average results
    plt.subplot(2, 1, 2)
    plt.plot(true_x, true_y, 'g-', label='True Path', linewidth=2)
    plt.scatter(measurements_x, measurements_y, c='r', marker='.', label='Noisy Measurements')
    plt.plot(ma_filtered_points[:, 0], ma_filtered_points[:, 1], 'm-', label=f'Moving Avg (w={ma_filter.window_size})', linewidth=2)
    plt.title('Moving Average Filter 2D Example')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    print("Moving Average filter test complete.")

    plt.tight_layout()
    plt.show()
    print("\nAll filter tests complete.") 