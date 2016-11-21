-- Author: Sixin Zhang (sixin.zhang@ens.fr)
-- adapted from nn.Sum

local opMean2d, parent = torch.class('scatwave.opMean2d', 'nn.Module')

function opMean2d:__init(dimension, nInputDims, sizeAverage)
   parent.__init(self)
   self.dimension   = dimension or 1
   -- do not assign default value to nInputDims or it will break backward compatibility
   self.nInputDims  = nInputDims
   self.sizeAverage = sizeAverage or true -- set false for non-averaging
   self.output = torch.Tensor()
end

function opMean2d:_getPositiveDimension(input)
    local dimension = self.dimension
    if dimension < 0 then
        dimension = input:dim() + dimension + 1
    elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
        dimension = dimension + 1
    end
    assert(input:dim() >= dimension, "dimension exceeds input dimensions")
    return dimension
end

function opMean2d:updateOutput(input)
    local dimension = self.dimension -- self:_getPositiveDimension(input)
    self.output:sum(input, dimension)
    if self.sizeAverage then
        self.output:div(input:size(dimension))
    end
    return self.output
end

function opMean2d:updateGradInput(input, gradOutput)
    local dimension = self.dimension  -- local dimension = self:_getPositiveDimension(input)
    -- zero-strides don't work with MKL/BLAS, so
    -- don't set self.gradInput to zero-stride tensor.
    -- Instead, do a deepcopy
    local size      = input:size()
    size[dimension] = 1
    if not gradOutput:isContiguous() then
        self._gradOutput = self._gradOutput or gradOutput.new()
        self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
        gradOutput = self._gradOutput
    end
    gradOutput      = gradOutput:view(size)
    self.gradInput:resizeAs(input)
    self.gradInput:copy(gradOutput:expandAs(input))
    if self.sizeAverage then
        self.gradInput:div(input:size(dimension))
    end
    return self.gradInput
end

function opMean2d:clearState()
    nn.utils.clear(self, '_gradOutput')
    return parent.clearState(self)
end
