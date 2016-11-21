require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

function pometicket(sec)
   local sec = sec or 0
   local t = os.date("*t")
   local ts = string.format('%d%.2d%.2d_%.2d%.2d%.2d',
                            t.year, t.month, t.day, t.hour, t.min, t.sec + sec)
   local hs = io.popen('hostname -s'):read()
   return hs .. '_' .. ts
end

function print_r ( t )  
    local print_r_cache={}
    local function sub_print_r(t,indent)
        if (print_r_cache[tostring(t)]) then
            print(indent.."*"..tostring(t))
        else
            print_r_cache[tostring(t)]=true
            if (type(t)=="table") then
                for pos,val in pairs(t) do
                    if (type(val)=="table") then
                        io.write(indent.."["..pos.."] => "..tostring(t).." {")
						io.write('\n')
                        sub_print_r(val,indent..string.rep(" ",string.len(pos)+8))
                        io.write(indent..string.rep(" ",string.len(pos)+6).."}")
						io.write('\n')
                    elseif (type(val)=="string") then
                        io.write(indent.."["..pos..'] => "'..val..'"')
						io.write('\n')
                    else
                        io.write(indent.."["..pos.."] => "..tostring(val))
						io.write('\n')
                    end
                end
            else
                io.write(indent..tostring(t))
				io.write('\n')
            end
        end
    end
    if (type(t)=="table") then
        io.write(tostring(t).." {")
		io.write('\n')
        sub_print_r(t,"  ")
        io.write("}")
		io.write('\n')
    else
        sub_print_r(t,"  ")
    end
	io.write('\n')
end
