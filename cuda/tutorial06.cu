#define MAX_NUL_LISTS 512
/* CPU host code*/
u32 find_min(const u32 * const src_arry,
            u32 * const list_indexes,
            const u32 num_lists,
            const u32 num_elements_per_list)
{
    u32 min_val = 0xFFFFFFFF;
    u32 min_idx = 0;
    //Iterator over each of the lists
    for(u32 i = 0; i < num_lists; i++)
    {
        //If the current list has already been emptied
        //the ignore it
        if(list_indexes[i] < num_elements_per_list)
        {
            const u32 src_idx = i + (list_indexes[i] * num_lists);
            const u32 data = src_array[src_idx];
            if(data <= min_val)
            {
                min_val = data;
                min_idx = i;
            }
        }
    }

    list_indexes[min_idx]++;
    return min_val;
}

void merge_array(const u32 * const_ src_array,
                u32 * const dest_array,
                const u32 num_lists, /*list number related to n threads*/
                const u32 num_elements)
{
    const u32 num_elements_per_list = (num_elements / num_lists);
    u32 list_indexes[MAX_NUL_LISTS];
    for(u32 list = 0; list < num_lists; list++)
    {
        list_indexes[list] = 0;
    }

    for(u32 i = 0; i < num_elements; i++)
    {
        dest_array[i] = find_min(src_array, list_indexes, num_lists, num_elements_per_list);
    }
}

/*GPU Heterogeneous code*/
__device__ void copy_data_to_shared(const u32 * const data, u32 * const sort_tmp
                                    const u32 num_lists, const u32 num_elements
                                    cosnt u32 tid)
{
    //Copy data into temp source
    for(u32 i = 0; i < num_elements; i += num_lists)
    {
        sort_tmp[i + tid] = data[i + tid];
    }
    __syncthreads();
}

/*Use a single thread for merge*/
void merge_array(const u32 * const_ src_array,
    u32 * const dest_array,
    const u32 num_lists, /*list number related to n threads*/
    const u32 num_elements)
{
    __shared__ u32 list_indexes[MAX_NUL_LISTS];

    //Multiple threads
    list_indexes[tid] = 0;
    __syncthreads();

    //Single threaded
    if(tid == 0)
    {
        const u32 num_elements_per_list = (num_elements / num_lists);
        for(u32 i = 0; i < num_elements; i++)
        {
            u32 min_val = 0xFFFFFFFF;
            u32 min_idx = 0;

            //Iterate over each of the lists
            for(u32 list = 0; list < num_lists; list++)
            {
                //If the current list has already benn emptied then ignore it
                if(list_indexes[list] < num_elements_per_list)
                {
                    const u32 src_idx = list + (list_indexes[list] * num_lists);
                    const u32 data = src_array[src_idx];
                    if(data <= min_val)
                    {
                        min_val = data;
                        min_idx = list;
                    }
                }
            }

            list_indexes[min_idx]++;
            dest_array[i] = min_val;
        }
    }
}

// Uses multiple threads for merge
// Deals with multiple indentical entries in the data
__device__ void merge_array6(const u32 * const src_array,
                            u32 * const dest_array,
                            const u32 num_lists,
                            const u32 num_elements,
                            const u32 tid)
{
    const u32 num_elements_per_list = (num_elements / num_lists);
    __shared__ u32 list_indexes[MAX_NUL_LISTS];
    list_indexes[tid] = 0;

    //Wait for list_indexes[tid] to be cleard
    __syncthreads();

    //Iterator over all elements
    for(u32 i = 0; i < num_elements; i++)
    {
        //Create a value shared with the other threads
        __shared__ u32 min_val;
        __shared__ u32 min_tid;

        //Use a temp register for work purposes
        u32 data;

        //If the current  list has no already benn emptied then
        //read from it, else ignore it
        if(list_indexes[tid] < num_elements_per_list)
        {
            //Work out from the list_index, the index into the linear array
            const u32 src_idx = tid + (list_indexes[tid] * num_lists);

            //Read the data from the list for given thread
            data = src_array[src_idx];
        }
        else
        {
            data = 0xFFFFFFFF;
        }

        //Have thread zero clear the min values
        if(tid == 0)
        {
            //Write a very large value so the first thread thread wins the min
            min_val = 0xFFFFFFFF;
            min_tid = 0xFFFFFFFF;
        }

        //Wait for all threads
        __syncthreads();

        //Hava every thread try to sotre it's value into min_val. Only the thread
        //with the lowest value will win
        atomicMin(&min_val, data);

        //Make sure all thread have taken their turn.
        __syncthreads();

        //If this thread was the one with the minium
        if(min_val == data)
        {
            //Check for equal values
            //Lowest tid wins and does the write
            atomicMin(&min_tid, tid);
        }

        //Make sure all threads have taken their turn.
        __syncthreads();

        //If this thread has the lowest tid
        if(tid == min_tid)
        {
            //Increment the list pointer for this thread
            list_indexes[tid]++;
            //Store the winning value
            dest_array[i] = data;
        }
    }
}

__global__ void gpu_sort_array_array(u32 * const data, const u32 num_lists,
                                    const u32 num_elements)
{
    const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ u32 sort_tmp[NUM_ELEM];
    __shared__ u32 sort_tmp_1[NUM_ELEM];

    copy_data_to_shared(data, sort_tmp, num_lists, num_elements, tid);
    radix_sort2(sort_tmp, num_lists, num_elements, tid, sort_tmp_1);
    merge_array6(sort_tmp, data, num_lists, num_elements, tid);
}