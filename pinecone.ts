import axios from "axios";

//  const getPineconeClient = () => {
if (!process.env.PINECONE_API_KEY) {
  throw new Error("PINECONE_API_KEY is not set");
}

const client = axios.create({
  baseURL: "https://atlasj-test-30db765.svc.us-east1-gcp.pinecone.io",
  headers: {
    "Content-Type": "application/json",
    "Api-Key": process.env.PINECONE_API_KEY,
  },
});

//  }

// wirte upsert function
export const upsert = async (id: string, vector: number[], metadata?: any) => {
  const response = await client.post(`/vectors/upsert`, {
    vectors: [
      {
        id,
        values: vector,
        metadata,
      },
    ],
    namespace: "",
  });
  return response.data;
};

interface QueryOptions {
    topK: number;
    includeMetadata: boolean;
    includeValues: boolean;
    namespace: string;
}
export const query = async  (vector:number[], options?:QueryOptions) => { 

    const topK = options?.topK ?? 3;
    const includeMetadata = options?.includeMetadata ?? false;
    const includeValues = options?.includeValues ?? false;
    const namespace = options?.namespace ?? "";
    
    const response = await client.post(`/query`, {
        vector,
        topK,
        includeMetadata,
        includeValues,
        namespace
    })

    return response.data;

 }





        
    
