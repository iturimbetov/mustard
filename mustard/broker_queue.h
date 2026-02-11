#pragma once

#include <cuda_runtime.h>

// Consistency mode for the broker queue:
//   FENCE  = nvshmem_fence() — orders operations to same PE (faster)
//   QUIET  = nvshmem_quiet() — ensures all pending operations complete globally (stronger)
enum class BrokerConsistency { FENCE, QUIET };

// Lock-free concurrent queue using NVSHMEM atomics.
// Requires NVSHMEM because updates can be remote.
template <BrokerConsistency Consistency = BrokerConsistency::QUIET>
class BrokerWorkDistributorT
{
public:
	typedef unsigned int Ticket;

	volatile Ticket *tickets;
	unsigned int *ring_buffer;
	int N;

	unsigned int *head;
	unsigned int *tail;
	int *count;

	BrokerWorkDistributorT(int size)
	{
		N = size;
		tickets = (volatile Ticket *)nvshmem_malloc(sizeof(Ticket) * N);
		ring_buffer = (unsigned int *)nvshmem_malloc(sizeof(unsigned int) * N);
		count = (int *)nvshmem_malloc(sizeof(int));
		head = (unsigned int *)nvshmem_malloc(sizeof(unsigned int));
		tail = (unsigned int *)nvshmem_malloc(sizeof(unsigned int));
	}

	void free_mem()
	{
		nvshmem_free((void *)tickets);
		nvshmem_free(ring_buffer);
		nvshmem_free(count);
		nvshmem_free(head);
		nvshmem_free(tail);
	}

	__device__ static __forceinline__ void backoff()
	{
		nvshmem_fence();
	}

	__forceinline__ __device__ void waitForTicket(const unsigned int P, const Ticket number, int pe)
	{
		while ((nvshmem_uint_g((unsigned int *)&tickets[P], pe)) != number)
		{
			backoff();
		}
	}

	__forceinline__ __device__ bool ensureDequeue(int pe)
	{
		int Num = nvshmem_int_atomic_fetch(count, pe);
		bool ensurance = false;

		while (!ensurance && Num > 0)
		{
			if (nvshmem_int_atomic_fetch_add(count, -1, pe) > 0)
			{
				ensurance = true;
			}
			else
			{
				Num = nvshmem_int_atomic_fetch_add(count, 1, pe) + 1;
			}
		}
		return ensurance;
	}

	__forceinline__ __device__ bool ensureEnqueue(int pe)
	{
		int Num = nvshmem_int_atomic_fetch(count, pe);
		bool ensurance = false;

		while (!ensurance && Num < N)
		{
			if (nvshmem_int_atomic_fetch_add(count, 1, pe) < N)
			{
				ensurance = true;
			}
			else
			{
				Num = nvshmem_int_atomic_fetch_add(count, -1, pe) - 1;
			}
		}
		return ensurance;
	}

	__forceinline__ __device__ void readData(unsigned int &val, int pe)
	{
		const unsigned int Pos = nvshmem_uint_atomic_fetch_inc(head, pe);
		const unsigned int P = Pos % N;

		waitForTicket(P, 2 * (Pos / N) + 1, pe);
		val = nvshmem_uint_g(&ring_buffer[P], pe);

		if constexpr (Consistency == BrokerConsistency::QUIET)
			nvshmem_quiet();
		else
			nvshmem_fence();

		nvshmem_uint_p((unsigned int *)&tickets[P], 2 * ((Pos + N) / N), pe);
	}

	__forceinline__ __device__ void putData(const unsigned int data, int pe)
	{
		const unsigned int Pos = nvshmem_uint_atomic_fetch_inc(tail, pe);
		const unsigned int P = Pos % N;
		const unsigned int B = 2 * (Pos / N);

		waitForTicket(P, B, pe);
		nvshmem_uint_p(&ring_buffer[P], data, pe);
		nvshmem_fence();
		nvshmem_uint_p((unsigned int *)&tickets[P], B + 1, pe);
	}

	__device__ inline bool enqueue(const unsigned int &data, int pe)
	{
		bool writeData = ensureEnqueue(pe);
		if (writeData)
		{
			putData(data, pe);
		}
		return false;
	}

	__device__ inline void dequeue(bool &hasData, unsigned int &data, int pe)
	{
		hasData = ensureDequeue(pe);
		if (hasData)
		{
			readData(data, pe);
		}
	}

	__device__ int size(int pe) const
	{
		return nvshmem_int_atomic_fetch(count, pe);
	}
};

using BrokerWorkDistributor = BrokerWorkDistributorT<BrokerConsistency::QUIET>;
using BrokerWorkDistributorFence = BrokerWorkDistributorT<BrokerConsistency::FENCE>;
