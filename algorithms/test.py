"""
        self.flows: List[Flow] = [
        #m_fid=1    125B 1000us 20ms
            Flow(1, 1, int(1e3), int(1e6), int(2e7),5, [7, 8], [15, 1, 6, 13]),
            Flow(1, 2, int(1e3), int(1e6), int(2e7),5, [7, 8], [15, 2, 10, 13]),
            Flow(1, 3, int(1e3), int(1e6), int(2e7),5, [7, 8], [15, 1, 6, 14]),
            Flow(1, 4, int(1e3), int(1e6), int(2e7),5, [7, 8], [15, 2, 10, 14]),
        #m_fid=2    125B 1000us 20ms
            Flow(2, 1, int(1e3), int(1e6), int(2e7),6, [5],  [16, 6, 12, 8, 3]),
            Flow(2, 2, int(1e3), int(1e6), int(2e7),6, [5],  [16, 5, 8, 3]),
        #m_fid=3    125B 1000us 20ms
            Flow(3, 1, int(1e3), int(1e6), int(2e7), 7, [5, 6], [17, 11, 5, 8, 3]),
            Flow(3, 2, int(1e3), int(1e6), int(2e7), 7, [5, 6], [17, 12, 9, 4, 3]),
            Flow(3, 3, int(1e3), int(1e6), int(2e7), 7, [5, 6], [17, 11, 4, 3]),
            Flow(3, 5, int(1e3), int(1e6), int(2e7), 7, [5, 6], [17, 11, 8, 3]),
            Flow(3, 6, int(1e3), int(1e6), int(2e7), 7, [5, 6], [17, 12, 9, 7]),
        #m_fid=4    200B 1000us 50ms
            Flow(4, 1, int(1.6e3), int(1e6), int(5e7), 8, [5], [18, 11, 5, 8, 3]),
            Flow(4, 2, int(1.6e3), int(1e6), int(5e7), 8, [5], [18, 12, 9, 4, 3]),
        #m_fid=5    300B 1000us 50ms
            Flow(5, 1, int(2.4e3), int(1e6), int(5e7), 6, [7, 8],  [16, 5, 10, 13]),
            Flow(5, 2, int(2.4e3), int(1e6), int(5e7), 6, [7, 8],  [16, 6, 14]),
        #m_fid=6    300B 1500us 50ms
            Flow(6, 1, int(2.4e3), int(1.5e6), int(5e7), 7, [5, 6], [17, 11, 5, 8, 3]),
            Flow(6, 2, int(2.4e3), int(1.5e6), int(5e7), 7, [5, 6], [17, 12, 9, 7]),
        #m_fid=7    625B 1500us 100ms
            Flow(7, 1, int(5e3), int(1.5e6), int(1e8), 5, [8], [15, 1, 6, 14]),
            Flow(7, 2, int(5e3), int(1.5e6), int(1e8), 5, [8], [15, 2, 10, 14]),
        #m_fid=8    625B 1500us 100ms
            Flow(8, 1, int(5e3), int(1.5e6), int(1e8), 6, [8],  [16, 5, 10, 14]),
            Flow(8, 2, int(5e3), int(1.5e6), int(1e8), 6, [8],  [16, 6, 14]),
        #m_fid=9    150B 3000us 100ms
            Flow(9, 1, int(1.2e3), int(3e6), int(1e8), 8, [5, 6], [18, 11, 5, 8, 3]),
            Flow(9, 2, int(1.2e3), int(3e6), int(1e8), 8, [5, 6], [18, 12, 9, 7]),
        #m_fid=10   150B 3000us 100ms
            Flow(10, 1, int(1.2e3), int(3e6), int(1e8), 8, [5], [18, 11, 5, 8, 3]),
            Flow(10, 2, int(1.2e3), int(3e6), int(1e8), 8, [5], [18, 12, 9, 4, 3])

        ]
"""
import matplotlib.pyplot as plt
import numpy as np
x = [50,100,150]
y = [2.59087,5,10]
plt.plot(x,y)

