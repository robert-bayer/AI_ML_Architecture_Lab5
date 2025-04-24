## Lab 5 Part 1 Discussion
Please answer the following questions.

### Your Name
Robert Bayer

### Briefly Explain What is MSCCLang-DSL. [0.5 points]
MSCCLang-DSL is a spcific language for MSCCLang that allows for efficient routing on chunks through collections of GPUs allowing for flexible routing algorithms

### How Many Types of Buffers Each GPU Have in MSCCLang? Which are They? [0.5 points]
- Number of Buffer Types: 3
- They are: Input, Output, Scratch

### Brifely Explain What Each Core Operations in MSCCLang-DSL Denotes. [0.5 points]
- `chunk`: returns a reference or handle to the chunks in the passed in buffer
- `copy`: copies chunks from the called on chunk object to the destiation passed in, returning a new reference or handle
- `reduce`: reduces the called on chunk and the passed in chunk in-place into the called on chunk. Returns a new refence or handle

### What is RecvReduceCopy Operation? [0.5 points]
An instruction DAG that receives a chunk and reduces it with the passed in source and then copies it to the passed in destination,

### Which Collective Algorithm the First Code Snippet Captures? Why Do you think So? [0.5 points]
- Algorithm: One-to-all Broadcast  
- Reason: The destination will be all other possible nodes except the same node

### Which Collective Algorithm the Second Code Snippet Captures? Why Do you think So? [0.5 points]
- Algorithm: All Gather (Ring)
- Reason: Each rank in the ring sends the chunk to the next rank interating all ranks in sequential order.
